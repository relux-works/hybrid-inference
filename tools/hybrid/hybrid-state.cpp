#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "hash/xxhash/xxhash.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(__APPLE__)
#define HYBRID_SHA_NI
#include <cpuid.h>
#include <immintrin.h>
#endif
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#endif

namespace hybrid {
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

json parse_json(const std::string & text, size_t max_bytes) {
    require(text.size() <= max_bytes, "JSON too large");
    return json::parse(text, [](int depth, json::parse_event_t, json &) {
        require(depth <= 32, "JSON nesting too deep");
        return true;
    });
}

static std::string hex(const unsigned char * p, size_t n) {
    static const char chars[] = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        out += chars[p[i] >> 4];
        out += chars[p[i] & 15];
    }
    return out;
}

std::string layout_fingerprint(const json & layout) {
    const nlohmann::json canonical = layout;
    const auto text = canonical.dump();
    unsigned char digest[32];
    sha256_hash(digest, reinterpret_cast<const unsigned char *>(text.data()), text.size());
    return hex(digest, sizeof(digest));
}

json model_layout(const llama_model * model) {
    const auto & hp = model->hparams;
    const auto & vocab = model->vocab;
    sha256_t hash;
    sha256_init(&hash);
    // Length framing preserves token boundaries and embedded NUL bytes, in token ID order.
    const int32_t n_tokens = llama_vocab_n_tokens(&vocab);
    for (llama_token i = 0; i < n_tokens; ++i) {
        const auto & token = vocab.get_token_data(i);
        uint64_t size = token.text.size();
        unsigned char length[8];
        for (int j = 0; j < 8; ++j) { length[j] = (unsigned char) (size >> (8 * j)); }
        sha256_update(&hash, length, sizeof(length));
        sha256_update(&hash, reinterpret_cast<const unsigned char *>(token.text.data()), token.text.size());
    }
    unsigned char digest[32];
    sha256_final(&hash, digest);
    json layers = json::array();
    for (uint32_t i = 0; i < hp.n_layer(); ++i) {
        layers.push_back({{"heads", hp.n_head(i)}, {"kv_heads", hp.n_head_kv(i)},
                          {"key_dim", hp.n_embd_head_k(i)}, {"value_dim", hp.n_embd_head_v(i)},
                          {"recurrent", hp.is_recr(i)}});
    }
    return {{"version", 1}, {"architecture", model->arch_name()}, {"layer_count", hp.n_layer()},
            {"embedding_size", hp.n_embd}, {"layers", layers},
            {"recurrent_layout", {{"conv", hp.ssm_d_conv}, {"inner", hp.ssm_d_inner},
                                  {"state", hp.ssm_d_state}, {"rank", hp.ssm_dt_rank}, {"groups", hp.ssm_n_group}}},
            {"vocab_size", vocab.n_tokens()}, {"vocab_sha256", hex(digest, sizeof(digest))}};
}

#ifdef HYBRID_SHA_NI
static bool sha_ni_available() {
    unsigned int a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || (c & bit_SSSE3) == 0 || (c & bit_SSE4_1) == 0) {
        return false;
    }
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & bit_SHA) != 0;
}

__attribute__((target("sha,ssse3,sse4.1")))
static void sha256_blocks_ni(uint32_t * state, const unsigned char * data, size_t blocks) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    const __m128i swap = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m128i abcd = _mm_loadu_si128(reinterpret_cast<const __m128i *>(state));
    __m128i efgh = _mm_loadu_si128(reinterpret_cast<const __m128i *>(state + 4));
    abcd = _mm_shuffle_epi32(abcd, 0xb1);
    efgh = _mm_shuffle_epi32(efgh, 0x1b);
    // SHA rounds use ABEF and CDGH, with the least significant word first.
    __m128i abef = _mm_alignr_epi8(abcd, efgh, 8);
    __m128i cdgh = _mm_blend_epi16(efgh, abcd, 0xf0);
    for (size_t block = 0; block < blocks; ++block, data += 64) {
        const __m128i saved_abef = abef;
        const __m128i saved_cdgh = cdgh;
        __m128i words[4];
        for (int i = 0; i < 4; ++i) {
            words[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data + 16 * i)), swap);
        }
        for (int i = 0; i < 16; ++i) {
            const int j = i & 3;
            if (i >= 4) {
                __m128i next = _mm_sha256msg1_epu32(words[j], words[(j + 1) & 3]);
                next = _mm_add_epi32(next, _mm_alignr_epi8(words[(j + 3) & 3], words[(j + 2) & 3], 4));
                words[j] = _mm_sha256msg2_epu32(next, words[(j + 3) & 3]);
            }
            __m128i rounds = _mm_add_epi32(words[j], _mm_loadu_si128(reinterpret_cast<const __m128i *>(k + 4 * i)));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, rounds);
            rounds = _mm_shuffle_epi32(rounds, 0x0e);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, rounds);
        }
        abef = _mm_add_epi32(abef, saved_abef);
        cdgh = _mm_add_epi32(cdgh, saved_cdgh);
    }
    abcd = _mm_shuffle_epi32(abef, 0x1b);
    efgh = _mm_shuffle_epi32(cdgh, 0xb1);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(state), _mm_blend_epi16(abcd, efgh, 0xf0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(state + 4), _mm_alignr_epi8(efgh, abcd, 8));
}

static void sha256_update_ni(sha256_t * hash, const unsigned char * data, size_t size) {
    const size_t pending = hash->count & 63;
    hash->count += size;
    if (pending) {
        const size_t copied = std::min(size, 64 - pending);
        std::memcpy(hash->buffer + pending, data, copied);
        data += copied;
        size -= copied;
        if (pending + copied < 64) { return; }
        sha256_blocks_ni(hash->state, hash->buffer, 1);
    }
    if (size >= 64) {
        sha256_blocks_ni(hash->state, data, size / 64);
        data += size & ~size_t(63);
        size &= 63;
    }
    if (size) { std::memcpy(hash->buffer, data, size); }
}
#endif

std::string file_hash(const std::string & path, size_t read_bytes, const char ** backend) {
    require(read_bytes > 0 && read_bytes <= 65536, "invalid hash read size");
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open model");
    const char * force_portable = std::getenv("HYBRID_HASH_PORTABLE");
    const bool portable = force_portable && std::strcmp(force_portable, "1") == 0;
    char buffer[65536];
    unsigned char digest[32];
#ifdef __APPLE__
    if (!portable) {
        if (backend) { *backend = "commoncrypto"; }
        CC_SHA256_CTX hash;
        require(CC_SHA256_Init(&hash) == 1, "cannot initialize model hash");
        while (file.read(buffer, read_bytes) || file.gcount()) {
            require(CC_SHA256_Update(&hash, buffer, static_cast<CC_LONG>(file.gcount())) == 1, "cannot update model hash");
        }
        require(file.eof(), "cannot read model");
        require(CC_SHA256_Final(digest, &hash) == 1, "cannot finalize model hash");
        return hex(digest, sizeof(digest));
    }
#endif
    auto update = sha256_update;
    const char * selected = "portable";
#ifdef HYBRID_SHA_NI
    if (!portable && sha_ni_available()) {
        update = sha256_update_ni;
        selected = "sha-ni";
    }
#else
    (void) portable;
#endif
    if (backend) { *backend = selected; }
    sha256_t hash;
    sha256_init(&hash);
    while (file.read(buffer, read_bytes) || file.gcount()) {
        update(&hash, reinterpret_cast<const unsigned char *>(buffer), file.gcount());
    }
    require(file.eof(), "cannot read model");
    // The vendor finalizer consumes the same state, byte count and partial block.
    sha256_final(&hash, digest);
    return hex(digest, sizeof(digest));
}

std::string prefix_hash(const std::vector<llama_token> & tokens) {
    sha256_t hash;
    sha256_init(&hash);
    for (auto token : tokens) {
        uint32_t t = (uint32_t) token;
        unsigned char bytes[4] = {(unsigned char) t, (unsigned char) (t >> 8), (unsigned char) (t >> 16), (unsigned char) (t >> 24)};
        sha256_update(&hash, bytes, sizeof(bytes));
    }
    unsigned char digest[32];
    sha256_final(&hash, digest);
    return hex(digest, sizeof(digest));
}

std::string export_range(llama_context * ctx, llama_pos n0, llama_pos n1) {
    size_t size = llama_state_seq_get_data_range(ctx, nullptr, 0, 0, n0, n1, 0);
    require(size > 0, "cannot size range");
    std::string out(size, '\0');
    require(llama_state_seq_get_data_range(ctx, reinterpret_cast<uint8_t *>(&out[0]), out.size(), 0, n0, n1, 0) == size, "cannot export range");
    return out;
}

static std::string checksums(const std::string & body) {
    XXH128_canonical_t xxh;
    XXH128_canonicalFromHash(&xxh, XXH3_128bits(body.data(), body.size()));
    std::string out(reinterpret_cast<const char *>(&xxh), sizeof(xxh));
    unsigned char digest[32];
    sha256_hash(digest, reinterpret_cast<const unsigned char *>(body.data()), body.size());
    out.append(reinterpret_cast<const char *>(digest), sizeof(digest));
    return out;
}

static size_t header_budget(const json & header) {
    require(header.at("n0").is_number_integer() && header.at("n1").is_number_integer(), "invalid positions");
    int64_t n0 = header.at("n0").get<int64_t>();
    int64_t n1 = header.at("n1").get<int64_t>();
    require(n0 >= 0 && n1 > n0 && n1 <= INT32_MAX, "invalid positions");
    // Nonnegative int32 IDs need at most ten digits and one separator each.
    return std::min<uint64_t>(max_delta_header_bytes, 4096 + 11 * uint64_t(n1 - n0));
}

std::string encode_delta_header(json header, size_t state_bytes) {
    header["magic"] = "HYBD";
    header["version"] = 1;
    header["body_format"] = "llama-native-seq-v1";
    header["kv_type"] = "q8_0";
    header["state_dtype"] = "f32";
    header["sections"] = json::array({{{"name", "llama_seq_state"}, {"offset", 0}, {"bytes", state_bytes}}});
    const std::string text = header.dump();
    require(text.size() <= header_budget(header), "delta header exceeds range budget");
    require(state_bytes > 0 && state_bytes <= max_delta_body_bytes, "invalid delta size");
    return text;
}

std::string encode_delta(json header, const std::string & state) {
    const std::string text = encode_delta_header(std::move(header), state.size());
    uint32_t n = text.size();
    std::string out;
    for (int i = 0; i < 4; ++i) { out += (char) (n >> (i * 8)); }
    return out + text + state + checksums(state);
}

delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0) {
    require(wire.size() >= 4, "truncated delta length");
    uint32_t size = 0;
    for (int i = 0; i < 4; ++i) { size |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    require(size <= max_delta_header_bytes && size <= wire.size() - 4, "invalid delta header length");
    json h = parse_json(wire.substr(4, size));
    require(h.is_object() && h.at("magic") == "HYBD" && h.at("version").is_number_integer() && h.at("version") == 1, "unsupported delta version");
    require(h.at("body_format") == "llama-native-seq-v1" && h.at("kv_type") == "q8_0" && h.at("state_dtype") == "f32", "unsupported delta layout");
    require(h.at("model_sha256").is_string() && h.at("model_sha256") == model_sha256, "model hash mismatch");
    require(h.at("session_id").is_string() && !h.at("session_id").get<std::string>().empty(), "invalid session id");
    require(h.at("n0").is_number_integer() && h.at("n1").is_number_integer(), "invalid positions");
    require(n0 >= 0 && h.at("n0").get<int64_t>() == n0 && h.at("n1").get<int64_t>() > n0 && h.at("n1").get<int64_t>() <= INT32_MAX, "position gap or overlap");
    require(size <= header_budget(h), "delta header exceeds range budget");
    auto hash = h.at("prefix_hash_n1").get<std::string>();
    require(hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos, "invalid prefix hash");
    const auto & tokens = h.at("tokens");
    require(tokens.is_array() && tokens.size() == (uint64_t) (h.at("n1").get<int64_t>() - n0), "delta token count mismatch");
    for (const auto & token : tokens) {
        require(token.is_number_integer() && token.get<int64_t>() >= 0 && token.get<int64_t>() <= INT32_MAX, "invalid delta token");
    }
    if (n0 == 0) {
        require(prefix_hash(tokens.get<std::vector<llama_token>>()) == hash, "delta prefix hash mismatch");
    }
    const auto & sections = h.at("sections");
    require(sections.is_array() && sections.size() == 1, "unsupported section layout");
    const auto & section = sections[0];
    require(section.at("name") == "llama_seq_state" && section.at("offset") == 0 && section.at("bytes").is_number_unsigned(), "invalid section layout");
    uint64_t bytes = section.at("bytes").get<uint64_t>();
    require(bytes > 0 && bytes <= max_delta_body_bytes && wire.size() - 4 - size == bytes + 48, "truncated or trailing delta bytes");
    std::string state = wire.substr(4 + size, bytes);
    require(wire.substr(4 + size + bytes) == checksums(state), "delta checksum mismatch");
    return {std::move(h), std::move(state)};
}

context_ptr make_context(llama_model * model, uint32_t n_ctx) {
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_seq_max = 1;
    params.n_threads = 2;
    params.n_threads_batch = 2;
    params.type_k = GGML_TYPE_Q8_0;
    params.type_v = GGML_TYPE_Q8_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.offload_kqv = false;
    context_ptr ctx(llama_init_from_model(model, params));
    require(bool(ctx), "cannot create CPU q8_0 context");
    return ctx;
}

void prefill(llama_context * ctx, const std::vector<llama_token> & tokens, size_t n0) {
    for (size_t offset = 0; offset < tokens.size(); offset += 256) {
        const size_t n = std::min<size_t>(256, tokens.size() - offset);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = n0 + offset + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
        batch.n_tokens = n;
        int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(result == 0, "prefill failed");
    }
}

std::string save_state(llama_context * ctx) {
    std::string state(llama_state_seq_get_size(ctx, 0), '\0');
    require(!state.empty() && llama_state_seq_get_data(ctx, reinterpret_cast<uint8_t *>(&state[0]), state.size(), 0) == state.size(), "cannot save state");
    return state;
}

void load_state(llama_context * ctx, const std::string & state) {
    if (state.empty()) {
        llama_memory_clear(llama_get_memory(ctx), true);
    } else {
        require(llama_state_seq_set_data(ctx, reinterpret_cast<const uint8_t *>(state.data()), state.size(), 0) == state.size(), "cannot load state");
    }
}

llama_token greedy(llama_context * ctx, const llama_model * model) {
    float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "no logits at decode boundary");
    int32_t n = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return std::max_element(logits, logits + n) - logits;
}
}
