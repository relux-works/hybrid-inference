#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-io.h"
#include "llama-memory.h"
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

void require_supported_architecture(const llama_model * model) {
    require(model->arch == LLM_ARCH_QWEN35 || model->arch == LLM_ARCH_GEMMA4 || model->arch == LLM_ARCH_MUSE_GLIMMER,
            "unsupported hybrid architecture");
}

json model_layout(const llama_model * model) {
    require_supported_architecture(model);
    const auto & hp = model->hparams;
    const bool swa = model->arch == LLM_ARCH_GEMMA4 || model->arch == LLM_ARCH_MUSE_GLIMMER;
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
        if (swa) {
            const int32_t source = hp.has_kv(i) ? i : hp.n_layer_kv_from_start - (hp.is_swa(i) ? 2 : 1);
            layers.back()["sliding"] = hp.is_swa(i);
            layers.back()["kv_source"] = source;
        }
    }
    json layout = {{"version", 1}, {"architecture", model->arch_name()}, {"layer_count", hp.n_layer()},
            {"embedding_size", hp.n_embd}, {"layers", layers},
            {"recurrent_layout", {{"conv", hp.ssm_d_conv}, {"inner", hp.ssm_d_inner},
                                  {"state", hp.ssm_d_state}, {"rank", hp.ssm_dt_rank}, {"groups", hp.ssm_n_group}}},
            {"vocab_size", vocab.n_tokens()}, {"vocab_sha256", hex(digest, sizeof(digest))}};
    if (swa) {
        require(hp.swa_type == LLAMA_SWA_TYPE_STANDARD && hp.n_swa > 0, "unsupported hybrid sliding window");
        layout["sliding_window"] = hp.n_swa;
        layout["sliding_window_type"] = "standard";
    }
    return layout;
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

class sha256_stream {
public:
    sha256_stream() {
        const char * value = std::getenv("HYBRID_HASH_PORTABLE");
        const bool portable = value && std::strcmp(value, "1") == 0;
#ifdef __APPLE__
        accelerated = !portable;
        if (accelerated) {
            backend = "commoncrypto";
            require(CC_SHA256_Init(&apple) == 1, "cannot initialize SHA-256");
            return;
        }
#endif
#ifdef HYBRID_SHA_NI
        if (!portable && sha_ni_available()) {
            update_fn = sha256_update_ni;
            backend = "sha-ni";
        }
#else
        (void) portable;
#endif
        sha256_init(&hash);
    }

    void update(const void * data, size_t size) {
#ifdef __APPLE__
        if (accelerated) {
            // Callers feed bounded chunks; CommonCrypto takes a 32-bit length.
            require(size <= UINT32_MAX, "SHA-256 chunk too large");
            require(CC_SHA256_Update(&apple, data, static_cast<CC_LONG>(size)) == 1, "cannot update SHA-256");
            return;
        }
#endif
        update_fn(&hash, static_cast<const unsigned char *>(data), size);
    }

    void finish(unsigned char * digest) {
#ifdef __APPLE__
        if (accelerated) {
            require(CC_SHA256_Final(digest, &apple) == 1, "cannot finalize SHA-256");
            return;
        }
#endif
        sha256_final(&hash, digest);
    }

    const char * backend = "portable";
private:
    sha256_t hash;
    decltype(&sha256_update) update_fn = sha256_update;
#ifdef __APPLE__
    bool accelerated = false;
    CC_SHA256_CTX apple;
#endif
};

std::string file_hash(const std::string & path, size_t read_bytes, const char ** backend) {
    require(read_bytes > 0 && read_bytes <= 65536, "invalid hash read size");
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open model");
    sha256_stream hash;
    if (backend) { *backend = hash.backend; }
    char buffer[65536];
    while (file.read(buffer, read_bytes) || file.gcount()) {
        hash.update(buffer, file.gcount());
    }
    require(file.eof(), "cannot read model");
    unsigned char digest[32];
    hash.finish(digest);
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

std::string export_range(llama_context * ctx, llama_pos n0, llama_pos n1, bool fixed_state) {
    const auto flags = fixed_state ? 0 : LLAMA_STATE_SEQ_FLAGS_KV_ONLY;
    size_t size = llama_state_seq_get_data_range(ctx, nullptr, 0, 0, n0, n1, flags);
    require(size > 0, "cannot size range");
    std::string out(size, '\0');
    require(llama_state_seq_get_data_range(ctx, reinterpret_cast<uint8_t *>(&out[0]), out.size(), 0, n0, n1, flags) == size, "cannot export range");
    return out;
}

static std::string checksums(std::string_view body) {
    llama_io_timer timer("checksum", body.size());
    sha256_stream sha;
    if (timer.enabled) { std::fprintf(stderr, "hybrid_import sha256_backend=%s\n", sha.backend); }
    const auto delete_xxh = [](XXH3_state_t * p) { XXH3_freeState(p); };
    std::unique_ptr<XXH3_state_t, decltype(delete_xxh)> xxh(XXH3_createState(), delete_xxh);
    require(bool(xxh) && XXH3_128bits_reset(xxh.get()) == XXH_OK, "cannot initialize XXH3");
    // Both digests consume the same cache-resident chunk before moving to the next one.
    for (size_t offset = 0; offset < body.size();) {
        const size_t bytes = std::min<size_t>(65536, body.size() - offset);
        require(XXH3_128bits_update(xxh.get(), body.data() + offset, bytes) == XXH_OK, "cannot update XXH3");
        sha.update(body.data() + offset, bytes);
        offset += bytes;
    }
    XXH128_canonical_t canonical;
    XXH128_canonicalFromHash(&canonical, XXH3_128bits_digest(xxh.get()));
    std::string out(reinterpret_cast<const char *>(&canonical), sizeof(canonical));
    unsigned char digest[32];
    sha.finish(digest);
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

static void validate_chunk_header(const json & h, int64_t n0) {
    require(h.at("transfer_id").is_string() && !h.at("transfer_id").get<std::string>().empty() &&
            h.at("transfer_id").get<std::string>().size() <= 128, "invalid transfer id");
    for (const auto * key : {"transfer_n0", "transfer_n1", "chunk_index", "chunk_count"}) {
        require(h.at(key).is_number_integer() && h.at(key).get<int64_t>() >= 0 && h.at(key).get<int64_t>() <= INT32_MAX, "invalid chunk metadata");
    }
    const int64_t begin = h.at("transfer_n0"), end = h.at("transfer_n1");
    const int64_t index = h.at("chunk_index"), count = h.at("chunk_count");
    require(begin <= n0 && end >= h.at("n1").get<int64_t>() && count > 0 && count <= end - begin && index < count &&
            (index != 0 || begin == n0) && (index + 1 == count ? end == h.at("n1").get<int64_t>() : end > h.at("n1").get<int64_t>()), "invalid chunk boundary");
    require(h.at("fixed_state").is_boolean() && h.at("fixed_state").get<bool>() == (index + 1 == count), "fixed state must occur only in final chunk");
}

// Qwen35 range bodies keep native f32 metadata; only recurrent tensor payloads shrink.
std::string transcode_recurrent(std::string_view state, bool pack, bool fixed_state) {
    require(state.size() <= max_delta_body_bytes, "native state exceeds bf16 body cap");
    size_t at = 0;
    auto skip = [&](size_t n) {
        require(n <= state.size() - at, "truncated bf16 state");
        at += n;
    };
    auto number = [&](size_t n) {
        require(n <= state.size() - at, "truncated bf16 state");
        uint64_t value = 0;
        for (size_t i = 0; i < n; ++i) { value |= uint64_t((unsigned char) state[at++]) << (8 * i); }
        return value;
    };
    skip(8); // native magic and source sequence
    require(number(4) == 1, "bf16 requires one Qwen35 KV stream");
    const auto cells = number(4);
    require(cells > 0 && cells <= (state.size() - at) / 24, "invalid bf16 KV cell count");
    for (uint64_t i = 0; i < cells; ++i) {
        skip(4);
        require(number(4) == 1, "invalid bf16 KV sequence count");
        skip(16); // M-RoPE metadata and sequence ID
    }
    require(number(4) == 0, "bf16 requires row-major KV");
    const auto layers = number(4);
    require(layers > 0 && layers <= 1024, "invalid bf16 KV layers");
    skip(8); // rotation metadata
    for (uint64_t i = 0; i < 2 * layers; ++i) {
        require(number(4) == GGML_TYPE_Q8_0, "bf16 requires q8_0 attention");
        const auto row = number(8);
        require(row > 0 && row <= (state.size() - at) / cells, "truncated bf16 KV");
        skip(row * cells);
    }
    if (!fixed_state) {
        require(at == state.size(), "early bf16 recurrent state");
        return std::string(state);
    }
    require(number(4) == 1, "invalid bf16 recurrent cell count");
    skip(4);
    require(number(4) == 0 && number(4) == 0, "invalid bf16 recurrent metadata");
    const auto total_layers = number(4);
    require(total_layers > layers && total_layers <= 1024, "invalid bf16 recurrent layers");
    std::string out(state.substr(0, at));
    for (uint64_t i = 0; i < 2 * (total_layers - layers); ++i) {
        const size_t meta = at;
        require(number(4) == GGML_TYPE_F32, "bf16 requires f32 recurrent tensors");
        const auto row = number(8);
        require(row > 0 && row <= max_delta_body_bytes && row % 4 == 0, "invalid bf16 recurrent row");
        const size_t bytes = pack ? row : row / 2;
        require(bytes <= state.size() - at, "truncated or odd-length bf16 state");
        require(out.size() <= max_delta_body_bytes - 12 && row <= max_delta_body_bytes - 12 - out.size(), "expanded bf16 state too large");
        out.append(state.substr(meta, 12));
        for (size_t j = 0; j < bytes; j += pack ? 4 : 2) {
            if (pack) {
                uint32_t bits = (uint32_t) number(4);
                float input;
                std::memcpy(&input, &bits, sizeof(input));
                const uint16_t value = ggml_fp32_to_bf16(input).bits;
                out += char(value); out += char(value >> 8);
            } else {
                const auto value = number(2);
                out += '\0'; out += '\0'; out += char(value); out += char(value >> 8);
            }
        }
    }
    require(at == state.size(), "trailing or odd-length bf16 state");
    return out;
}

std::string encode_delta_header(json header, size_t state_bytes) {
    header["magic"] = "HYBD";
    header["version"] = 2;
    if (!header.contains("transfer_id")) {
        header["transfer_id"] = header.at("session_id");
        header["transfer_n0"] = header.at("n0");
        header["transfer_n1"] = header.at("n1");
        header["chunk_index"] = 0;
        header["chunk_count"] = 1;
        header["fixed_state"] = true;
    }
    header["body_format"] = "llama-native-chunks-v3";
    header["kv_type"] = "q8_0";
    const auto dtype = header.value("state_dtype", std::string("f32"));
    require(dtype == "f32" || dtype == "bf16", "unsupported state dtype");
    header["state_dtype"] = dtype;
    header["sections"] = json::array({{{"name", "llama_seq_state"}, {"offset", 0}, {"bytes", state_bytes}}});
    validate_chunk_header(header, header.at("n0").get<int64_t>());
    const std::string text = header.dump();
    require(text.size() <= header_budget(header), "delta header exceeds range budget");
    require(state_bytes > 0 && state_bytes <= max_delta_body_bytes, "invalid delta size");
    return text;
}

std::string encode_delta(json header, const std::string & state) {
    header_budget(header); // Reject empty ranges before dtype-specific parsing.
    require(!state.empty() && state.size() <= max_delta_body_bytes, "invalid delta size");
    const auto dtype = header.value("state_dtype", std::string("f32"));
    const auto compressed = dtype == "bf16" ? transcode_recurrent(state, true, header.value("fixed_state", true)) : std::string();
    const auto & body = dtype == "bf16" ? compressed : state;
    const std::string text = encode_delta_header(std::move(header), body.size());
    uint32_t n = text.size();
    std::string out;
    for (int i = 0; i < 4; ++i) { out += (char) (n >> (i * 8)); }
    out.reserve(4 + text.size() + body.size() + 48);
    out += text;
    out += body;
    out += checksums(body);
    return out;
}

static json read_delta_header(const std::string & wire, uint32_t & size) {
    require(wire.size() >= 4, "truncated delta length");
    size = 0;
    for (int i = 0; i < 4; ++i) { size |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    require(size <= max_delta_header_bytes && size <= wire.size() - 4, "invalid delta header length");
    return parse_json(wire.substr(4, size));
}

delta_view read_delta_view(const std::string & wire, const std::string & model_sha256, int64_t n0) {
    llama_io_timer timer("codec", wire.size());
    uint32_t size;
    json h = read_delta_header(wire, size);
    require(h.is_object() && h.at("magic") == "HYBD" && h.at("version").is_number_integer() && h.at("version") == 2, "unsupported delta version");
    require(h.at("body_format") == "llama-native-chunks-v3" && h.at("kv_type") == "q8_0" && (h.at("state_dtype") == "f32" || h.at("state_dtype") == "bf16"), "unsupported delta layout");
    require(h.at("model_sha256").is_string() && h.at("model_sha256") == model_sha256, "model hash mismatch");
    require(h.at("session_id").is_string() && !h.at("session_id").get<std::string>().empty(), "invalid session id");
    require(h.at("n0").is_number_integer() && h.at("n1").is_number_integer(), "invalid positions");
    require(n0 >= 0 && h.at("n0").get<int64_t>() == n0 && h.at("n1").get<int64_t>() > n0 && h.at("n1").get<int64_t>() <= INT32_MAX, "position gap or overlap");
    validate_chunk_header(h, n0);
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
    const std::string_view state(wire.data() + 4 + size, bytes);
    require(std::string_view(wire.data() + 4 + size + bytes, 48) == checksums(state), "delta checksum mismatch");
    return {std::move(h), state};
}

delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0, const std::string & expected_dtype) {
    auto view = read_delta_view(wire, model_sha256, n0);
    require(expected_dtype.empty() || view.header.at("state_dtype") == expected_dtype, "state dtype mismatch");
    llama_io_timer timer("state_copy", view.state.size());
    auto state = view.header.at("state_dtype") == "bf16" ? transcode_recurrent(view.state, false, view.header.at("fixed_state")) : std::string(view.state);
    return {std::move(view.header), std::move(state)};
}

void delta_import::clear() {
    chunks.clear();
    wires.clear();
    prefix.clear();
    tokens.clear();
}

bool delta_import::stage(llama_context * ctx, llama_seq_id seq_id, std::string wire, const std::string & model_hash, const std::string & session_id,
                         const std::vector<llama_token> & current, size_t n_ctx, int32_t n_vocab, const std::string & state_dtype) {
    try {
        if (!chunks.empty()) {
            uint32_t size;
            const auto h = read_delta_header(wire, size);
            if (h.at("chunk_index") == 0 && h.at("transfer_id") != chunks.front().header.at("transfer_id")) {
                clear();
            }
        }
        // Heap-owned wires keep borrowed body addresses stable as the staging vectors grow.
        auto source = std::make_unique<std::string>(std::move(wire));
        auto next = read_delta_view(*source, model_hash, chunks.empty() ? current.size() : tokens.size());
        require(next.header.at("state_dtype") == state_dtype, "state dtype mismatch");
        if (next.header.at("state_dtype") == "bf16") {
            auto expanded = std::make_unique<std::string>(transcode_recurrent(next.state, false, next.header.at("fixed_state")));
            next.state = *expanded;
            source = std::move(expanded);
        }
        const auto & h = next.header;
        require(h.at("session_id") == session_id, "Delta session mismatch");
        require(h.at("transfer_n1").get<uint64_t>() <= n_ctx, "Delta exceeds slot context");
        require(h.at("chunk_index").get<size_t>() == chunks.size(), "out-of-order or duplicate chunk");
        if (chunks.empty()) {
            require(h.at("transfer_n0").get<uint64_t>() == current.size(), "transfer prefix mismatch");
            prefix = current;
            tokens = current;
        } else {
            require(prefix == current, "slot changed during transfer");
            const auto & first = chunks.front().header;
            for (const auto * key : {"session_id", "transfer_id", "transfer_n0", "transfer_n1", "chunk_count", "state_dtype"}) {
                require(first.at(key) == h.at(key), "chunk transfer mismatch");
            }
        }
        const auto suffix = h.at("tokens").get<std::vector<llama_token>>();
        for (auto token : suffix) { require(token < n_vocab, "Delta token outside vocabulary"); }
        tokens.insert(tokens.end(), suffix.begin(), suffix.end());
        require(prefix_hash(tokens) == h.at("prefix_hash_n1"), "Delta prefix hash mismatch");
        const bool final = h.at("fixed_state");
        const llama_state_seq_chunk native {reinterpret_cast<const uint8_t *>(next.state.data()), next.state.size(), h.at("n1").get<llama_pos>()};
        require(llama_state_seq_validate_chunk(ctx, native, seq_id, h.at("n0").get<llama_pos>(), final) == next.state.size(), "invalid native chunk or fixed state");
        wires.push_back(std::move(source));
        chunks.push_back(std::move(next));
        return final;
    } catch (...) {
        clear();
        throw;
    }
}

void delta_import::apply(llama_context * ctx, llama_seq_id seq_id) {
    try {
        require(!chunks.empty() && chunks.back().header.at("fixed_state") == true, "missing final chunk");
        std::vector<llama_state_seq_chunk> native;
        size_t bytes = 0;
        for (const auto & chunk : chunks) {
            require(chunk.state.size() <= SIZE_MAX - bytes, "chunk bytes overflow");
            bytes += chunk.state.size();
            native.push_back({reinterpret_cast<const uint8_t *>(chunk.state.data()), chunk.state.size(), chunk.header.at("n1").get<llama_pos>()});
        }
        require(llama_state_seq_set_data_chunks(ctx, native, seq_id, prefix.size()) == bytes, "Native delta chunks rejected; slot preserved");
    } catch (...) {
        clear();
        throw;
    }
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
    params.swa_full = false;
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
    const auto flags = llama_get_model(ctx)->hparams.swa_type == LLAMA_SWA_TYPE_STANDARD ? LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT : 0;
    std::string state(llama_state_seq_get_size_ext(ctx, 0, flags), '\0');
    require(!state.empty() && llama_state_seq_get_data_ext(ctx, reinterpret_cast<uint8_t *>(&state[0]), state.size(), 0, flags) == state.size(), "cannot save state");
    return state;
}

void load_state(llama_context * ctx, const std::string & state) {
    if (state.empty()) {
        llama_memory_clear(llama_get_memory(ctx), true);
    } else {
        const auto flags = llama_get_model(ctx)->hparams.swa_type == LLAMA_SWA_TYPE_STANDARD ? LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT : 0;
        require(llama_state_seq_set_data_ext(ctx, reinterpret_cast<const uint8_t *>(state.data()), state.size(), 0, flags) == state.size(), "cannot load state");
    }
}

llama_token greedy(llama_context * ctx, const llama_model * model) {
    float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "no logits at decode boundary");
    int32_t n = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return std::max_element(logits, logits + n) - logits;
}
}
