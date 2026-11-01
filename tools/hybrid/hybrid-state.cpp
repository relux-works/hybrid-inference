#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-memory.h"
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

std::string export_range(llama_context * ctx, llama_pos n0, llama_pos n1, bool fixed_state) {
    const auto flags = fixed_state ? 0 : LLAMA_STATE_SEQ_FLAGS_KV_ONLY;
    size_t size = llama_state_seq_get_data_range(ctx, nullptr, 0, 0, n0, n1, flags);
    require(size > 0, "cannot size range");
    std::string out(size, '\0');
    require(llama_state_seq_get_data_range(ctx, reinterpret_cast<uint8_t *>(&out[0]), out.size(), 0, n0, n1, flags) == size, "cannot export range");
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
    header["state_dtype"] = "f32";
    header["sections"] = json::array({{{"name", "llama_seq_state"}, {"offset", 0}, {"bytes", state_bytes}}});
    validate_chunk_header(header, header.at("n0").get<int64_t>());
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

static json read_delta_header(const std::string & wire, uint32_t & size) {
    require(wire.size() >= 4, "truncated delta length");
    size = 0;
    for (int i = 0; i < 4; ++i) { size |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    require(size <= max_delta_header_bytes && size <= wire.size() - 4, "invalid delta header length");
    return parse_json(wire.substr(4, size));
}

delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0) {
    uint32_t size;
    json h = read_delta_header(wire, size);
    require(h.is_object() && h.at("magic") == "HYBD" && h.at("version").is_number_integer() && h.at("version") == 2, "unsupported delta version");
    require(h.at("body_format") == "llama-native-chunks-v3" && h.at("kv_type") == "q8_0" && h.at("state_dtype") == "f32", "unsupported delta layout");
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
    std::string state = wire.substr(4 + size, bytes);
    require(wire.substr(4 + size + bytes) == checksums(state), "delta checksum mismatch");
    return {std::move(h), std::move(state)};
}

void delta_import::clear() {
    chunks.clear();
    prefix.clear();
    tokens.clear();
}

bool delta_import::stage(llama_context * ctx, llama_seq_id seq_id, const std::string & wire, const std::string & model_hash, const std::string & session_id,
                         const std::vector<llama_token> & current, size_t n_ctx, int32_t n_vocab) {
    try {
        if (!chunks.empty()) {
            uint32_t size;
            const auto h = read_delta_header(wire, size);
            if (h.at("chunk_index") == 0 && h.at("transfer_id") != chunks.front().header.at("transfer_id")) {
                clear();
            }
        }
        auto next = read_delta(wire, model_hash, chunks.empty() ? current.size() : tokens.size());
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
            for (const auto * key : {"session_id", "transfer_id", "transfer_n0", "transfer_n1", "chunk_count"}) {
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
