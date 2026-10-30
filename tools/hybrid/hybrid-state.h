#pragma once

#include "llama.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace hybrid {
using json = nlohmann::ordered_json;

json parse_json(const std::string & text);
std::string prefix_hash(const std::vector<llama_token> & tokens);
std::string file_hash(const std::string & path);
std::string export_range(llama_context * ctx, llama_pos n0, llama_pos n1);
std::string encode_delta(json header, const std::string & state);
struct delta {
    json header;
    std::string state;
};
delta read_delta(const std::string & wire, const std::string & model_sha256, int64_t n0);

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
