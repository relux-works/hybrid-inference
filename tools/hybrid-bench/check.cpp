#include "hybrid-state.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::string read_file(const char * path, size_t limit = 513 * 1024 * 1024 + 52) {
    std::ifstream source(path, std::ios::binary);
    require(source.good(), "cannot read check input");
    source.seekg(0, std::ios::end);
    require(source.tellg() >= 0 && uint64_t(source.tellg()) <= limit, "check input exceeds limit");
    source.seekg(0);
    return {std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
}

static double compare(const float * p, const float * q, int n) {
    const double pmax = *std::max_element(p, p + n);
    const double qmax = *std::max_element(q, q + n);
    double psum = 0, qsum = 0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(p[i]) && std::isfinite(q[i]), "nonfinite logits");
        psum += std::exp(p[i] - pmax);
        qsum += std::exp(q[i] - qmax);
    }
    const double pz = pmax + std::log(psum), qz = qmax + std::log(qsum);
    double kl = 0;
    for (int i = 0; i < n; ++i) {
        kl += std::exp(p[i] - pz) * (p[i] - pz - q[i] + qz);
    }
    return std::max(0.0, kl);
}

static hybrid::context_ptr context(llama_model * model, int depth, bool gpu) {
    auto cp = llama_context_default_params();
    cp.n_ctx = depth;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = gpu;
    hybrid::context_ptr ctx(llama_init_from_model(model, cp));
    require(bool(ctx), "cannot create check context");
    return ctx;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 6, "usage: llama-hybrid-bench-check MODEL TOKENS.json DELTA OUTPUT.json GPU_LAYERS");
        // Full-vocabulary KL must detect a changed distribution, including one with unchanged top-1.
        const float p[] = {2, -2}, q[] = {-2, 2}, near[] = {1, 0};
        require(compare(p, p, 2) < 1e-12 && compare(p, q, 2) > .01 && compare(p, near, 2) > .01, "KL control failed");
        const auto fixture = hybrid::json::parse(read_file(argv[2], 8 * 1024 * 1024), [](int depth, hybrid::json::parse_event_t, hybrid::json &) {
            require(depth <= 32, "check JSON nesting too deep");
            return true;
        });
        const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
        const int k = fixture.at("k");
        const int gpu_layers = std::stoi(argv[5]);
        require(tokens.size() >= 2 && tokens.size() <= 1048576 && tokens.size() + k + 2 <= fixture.value("max_ctx", 16384) && k > 0 && k <= 256,
                "check depth or continuation exceeds prototype bounds");
        require(gpu_layers >= 0 && gpu_layers <= 999, "invalid GPU layers");
        const auto model_hash = hybrid::file_hash(argv[1]);
        require(model_hash == fixture.at("model_sha256"), "check model mismatch");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "cannot load check model");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) { require(token >= 0 && token < vocab, "invalid check token"); }
        const int ctx = ((tokens.size() + k + 257) / 256) * 256;
        auto imported = context(model.get(), ctx, gpu_layers > 0);
        auto split = context(model.get(), ctx, gpu_layers > 0);
        auto whole = context(model.get(), ctx, gpu_layers > 0);
        auto free_whole = context(model.get(), ctx, gpu_layers > 0);
        int64_t n = 0;
        const auto paths = fixture.value("deltas", std::vector<std::string>{argv[3]});
        require(!paths.empty(), "missing check deltas");
        for (const auto & path : paths) {
            const auto delta = hybrid::read_delta(read_file(path.c_str()), model_hash, n);
            const int64_t n1 = delta.header.at("n1");
            require(n1 <= int64_t(tokens.size() - 1), "check delta endpoint mismatch");
            require(delta.header.at("tokens").get<std::vector<llama_token>>() ==
                    std::vector<llama_token>(tokens.begin() + n, tokens.begin() + n1), "check delta tokens mismatch");
            require(delta.header.at("prefix_hash_n1") == hybrid::prefix_hash({tokens.begin(), tokens.begin() + n1}), "check prefix mismatch");
            require(llama_state_seq_set_data_append(imported.get(), reinterpret_cast<const uint8_t *>(delta.state.data()),
                    delta.state.size(), 0, n, n1) == delta.state.size(), "check delta import failed");
            hybrid::prefill(split.get(), {tokens.begin() + n, tokens.begin() + n1}, n);
            n = n1;
        }
        require(n == int64_t(tokens.size() - 1), "incomplete check chain");
        hybrid::prefill(imported.get(), {tokens.back()}, n);
        hybrid::prefill(split.get(), {tokens.back()}, n);
        hybrid::prefill(whole.get(), tokens, 0);
        hybrid::prefill(free_whole.get(), tokens, 0);
        double kl_sum = 0;
        int top1 = 0, same = 0, first_divergence = -1;
        for (int i = 0; i < k; ++i) {
            const auto token = hybrid::greedy(imported.get(), model.get());
            same += token == hybrid::greedy(split.get(), model.get());
            top1 += token == hybrid::greedy(whole.get(), model.get());
            const auto free_token = hybrid::greedy(free_whole.get(), model.get());
            if (first_divergence < 0 && token != free_token) { first_divergence = i; }
            kl_sum += compare(llama_get_logits_ith(imported.get(), -1), llama_get_logits_ith(whole.get(), -1), vocab);
            // Teacher forcing uses the imported continuation for both numerical references.
            hybrid::prefill(imported.get(), {token}, tokens.size() + i);
            hybrid::prefill(split.get(), {token}, tokens.size() + i);
            hybrid::prefill(whole.get(), {token}, tokens.size() + i);
            hybrid::prefill(free_whole.get(), {free_token}, tokens.size() + i);
        }
        const double kl = kl_sum / k, agreement = double(top1) / k;
        hybrid::json result = {{"stage", "equivalence"}, {"depth", tokens.size()}, {"k", k},
            {"same_partition_agreed", same}, {"top1_agreed", top1}, {"top1_agreement", agreement},
            {"mean_kl_nats", kl}, {"free_running_first_divergence", first_divergence},
            {"kl_limit", .01}, {"top1_minimum", .95}, {"same_partition_exact", same == k},
            {"numerical_within_bounds", kl <= .01 && agreement >= .95}};
        std::ofstream target(argv[4]);
        target << result.dump(2) << '\n';
        require(target.good(), "cannot write check result");
        std::cout << result.dump() << '\n';
        return same == k && kl <= .01 && agreement >= .95 ? 0 : 3;
    } catch (const std::exception & error) {
        std::cerr << "check failed: " << error.what() << '\n';
        return 2;
    }
}
