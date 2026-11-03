#pragma once

#include "llama.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hybrid {
using json = nlohmann::ordered_json;

json parse_json(const std::string & text, size_t max_bytes = 1024 * 1024);
std::string prefix_hash(const std::vector<llama_token> & tokens);
std::string file_hash(const std::string & path, size_t read_bytes = 65536, const char ** backend = nullptr);
void require_supported_architecture(const llama_model * model);
json model_layout(const llama_model * model);
std::string layout_fingerprint(const json & layout);
std::string export_range(llama_context * ctx, llama_pos n0, llama_pos n1, bool fixed_state = true);
constexpr size_t max_delta_header_bytes = 1024 * 1024;
constexpr size_t max_delta_body_bytes = 512 * 1024 * 1024;
std::string transcode_recurrent(std::string_view state, bool pack, bool fixed_state);
std::string encode_delta_header(json header, size_t state_bytes);
std::string encode_delta(json header, const std::string & state);
struct delta {
    json header;
    std::string state;
};
delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0, const std::string & expected_dtype = "");
struct delta_view {
    json header;
    std::string_view state;
};
// The view retains wire dtype. Use read_delta or delta_import for bf16 expansion.
// The wire must outlive this view and all tensor writes made from it.
delta_view read_delta_view(const std::string & wire, const std::string & model_sha256, int64_t n0);
delta_view read_delta_view(std::string && wire, const std::string & model_sha256, int64_t n0) = delete;

// A pending import owns only host bytes. The live slot changes in apply(), after the final chunk.
class delta_import {
    std::vector<std::unique_ptr<std::string>> wires;
    std::vector<delta_view> chunks;
    std::vector<llama_token> prefix;
    std::vector<llama_token> tokens;
public:
    void clear();
    bool stage(llama_context * ctx, llama_seq_id seq_id, std::string wire, const std::string & model_hash, const std::string & session_id,
               const std::vector<llama_token> & current, size_t n_ctx, int32_t n_vocab, const std::string & state_dtype = "f32");
    const std::vector<llama_token> & imported_tokens() const { return tokens; }
    void apply(llama_context * ctx, llama_seq_id seq_id);
};

struct model_deleter { void operator()(llama_model * p) const { llama_model_free(p); } };
struct context_deleter { void operator()(llama_context * p) const { llama_free(p); } };
using model_ptr = std::unique_ptr<llama_model, model_deleter>;
using context_ptr = std::unique_ptr<llama_context, context_deleter>;
context_ptr make_context(llama_model * model, uint32_t n_ctx);
void prefill(llama_context * ctx, const std::vector<llama_token> & tokens, size_t n0);
std::string save_state(llama_context * ctx);
void load_state(llama_context * ctx, const std::string & state);
llama_token greedy(llama_context * ctx, const llama_model * model);
}
