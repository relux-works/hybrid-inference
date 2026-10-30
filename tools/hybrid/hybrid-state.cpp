#include "hybrid-state.h"
#include "llama-ext.h"
#include "hash/xxhash/xxhash.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace hybrid {
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

json parse_json(const std::string & text) {
    require(text.size() <= 1024 * 1024, "JSON too large");
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

std::string file_hash(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open model");
    sha256_t hash;
    sha256_init(&hash);
    char buffer[65536];
    while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
        sha256_update(&hash, reinterpret_cast<const unsigned char *>(buffer), file.gcount());
    }
    require(file.eof(), "cannot read model");
    unsigned char digest[32];
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

std::string encode_delta(json header, const std::string & state) {
    header["magic"] = "HYBD";
    header["version"] = 1;
    header["body_format"] = "llama-native-seq-v1";
    header["kv_type"] = "q8_0";
    header["state_dtype"] = "f32";
    header["sections"] = json::array({{{"name", "llama_seq_state"}, {"offset", 0}, {"bytes", state.size()}}});
    const std::string text = header.dump();
    require(text.size() <= 65536 && !state.empty() && state.size() <= 512 * 1024 * 1024, "invalid delta size");
    uint32_t n = text.size();
    std::string out;
    for (int i = 0; i < 4; ++i) { out += (char) (n >> (i * 8)); }
    return out + text + state + checksums(state);
}

delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0) {
    require(wire.size() >= 4, "truncated delta length");
    uint32_t size = 0;
    for (int i = 0; i < 4; ++i) { size |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    require(size <= 65536 && size <= wire.size() - 4, "invalid delta header length");
    json h = parse_json(wire.substr(4, size));
    require(h.is_object() && h.at("magic") == "HYBD" && h.at("version").is_number_integer() && h.at("version") == 1, "unsupported delta version");
    require(h.at("body_format") == "llama-native-seq-v1" && h.at("kv_type") == "q8_0" && h.at("state_dtype") == "f32", "unsupported delta layout");
    require(h.at("model_sha256").is_string() && h.at("model_sha256") == model_sha256, "model hash mismatch");
    require(h.at("session_id").is_string() && !h.at("session_id").get<std::string>().empty(), "invalid session id");
    require(h.at("n0").is_number_integer() && h.at("n1").is_number_integer(), "invalid positions");
    require(n0 >= 0 && h.at("n0").get<int64_t>() == n0 && h.at("n1").get<int64_t>() > n0 && h.at("n1").get<int64_t>() <= INT32_MAX, "position gap or overlap");
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
    require(bytes > 0 && bytes <= 512 * 1024 * 1024 && wire.size() - 4 - size == bytes + 48, "truncated or trailing delta bytes");
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
