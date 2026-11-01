#include "hybrid-state.h"
#include "llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

static double max_difference(const float * p, const float * q, int n) {
    double result = 0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(p[i]) && std::isfinite(q[i]), "nonfinite logits");
        result = std::max(result, std::abs(double(p[i]) - q[i]));
    }
    return result;
}

static bool within_repeat_floor(bool exact, bool repeat_exact, double kl, double diff, double repeat_kl, double repeat_diff) {
    return repeat_exact ? exact : diff <= repeat_diff && kl <= repeat_kl;
}

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static hybrid::context_ptr context(llama_model * model, int depth, bool gpu) {
    auto cp = llama_context_default_params();
    cp.n_ctx = depth;
    cp.n_batch = cp.n_ubatch = 256;
    cp.swa_full = false;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = gpu;
    cp.op_offload = gpu;
    hybrid::context_ptr ctx(llama_init_from_model(model, cp));
    require(bool(ctx), "cannot create check context");
    return ctx;
}

static std::vector<size_t> boundaries(size_t depth, size_t chunk, size_t start = 0) {
    std::vector<size_t> ends;
    for (size_t n = start; n < depth;) {
        n = std::min(depth, n + chunk);
        ends.push_back(n);
    }
    return ends;
}

static void prefill_partition(llama_context * ctx, const std::vector<llama_token> & tokens, size_t chunk) {
    for (size_t n = 0; n < tokens.size(); n += chunk) {
        const size_t end = std::min(tokens.size(), n + chunk);
        hybrid::prefill(ctx, {tokens.begin() + n, tokens.begin() + end}, n);
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 6, "usage: llama-hybrid-bench-check MODEL TOKENS.json DELTA OUTPUT.json GPU_LAYERS");
        // Full-vocabulary KL must detect a changed distribution, including one with unchanged top-1.
        const float p[] = {2, -2}, q[] = {-2, 2}, near[] = {1, 0};
        require(compare(p, p, 2) < 1e-12 && compare(p, q, 2) > .01 && compare(p, near, 2) > .01, "KL control failed");
        // The measured rule accepts its observed bound, rejects either excess, and keeps exact references bit-exact.
        require(within_repeat_floor(true, true, 0, 0, 0, 0) && !within_repeat_floor(false, true, 0, 0, 0, 0) &&
                within_repeat_floor(false, false, .01, .1, .01, .1) && !within_repeat_floor(false, false, .02, .1, .01, .1) &&
                !within_repeat_floor(false, false, .01, .2, .01, .1), "measured floor control failed");
        const auto fixture = hybrid::json::parse(read_file(argv[2], 8 * 1024 * 1024), [](int depth, hybrid::json::parse_event_t, hybrid::json &) {
            require(depth <= 32, "check JSON nesting too deep");
            return true;
        });
        const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
        const int k = fixture.at("k");
        const int gpu_layers = std::stoi(argv[5]);
        const auto chunks = fixture.value("noise_chunks", std::vector<int>{128, 255});
        require(!chunks.empty() && chunks.size() <= 4, "need 1..4 noise chunks");
        for (int chunk : chunks) { require(chunk > 0 && chunk <= 256, "noise chunk must be in 1..256"); }
        require(tokens.size() >= 3 && tokens.size() <= 1048576 && tokens.size() + k + 2 <= fixture.value("max_ctx", 16384) && k > 0 && k <= 256,
                "check depth or continuation exceeds prototype bounds");
        require(gpu_layers >= 0 && gpu_layers <= 999, "invalid GPU layers");
        const auto model_hash = hybrid::file_hash(argv[1]);
        require(model_hash == fixture.at("model_sha256"), "check model mismatch");
        llama_backend_init();
        require(gpu_layers == 0 || llama_supports_gpu_offload(), "GPU requested but checker has no GPU offload device");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "cannot load check model");
        std::string backend = "cpu";
        if (gpu_layers > 0) {
            require(llama_model_n_devices(model.get()) > 0, "GPU requested but model has no offload device");
            const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(llama_model_get_device(model.get(), 0)));
            backend = name == "CUDA" ? "cuda" : name == "Metal" ? "metal" : name;
            for (int i = 0; i < llama_model_n_devices(model.get()); ++i) {
                require(name == ggml_backend_reg_name(ggml_backend_dev_backend_reg(llama_model_get_device(model.get(), i))), "mixed checker devices unsupported");
            }
        }
        require(fixture.value("backend", backend) == backend, "checker backend label/device mismatch");
        const bool gpu = backend != "cpu";
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) { require(token >= 0 && token < vocab, "invalid check token"); }
        const int ctx = ((tokens.size() + k + 257) / 256) * 256;
        auto imported = context(model.get(), ctx, gpu);
        auto split = context(model.get(), ctx, gpu);
        auto whole = context(model.get(), ctx, gpu);
        auto free_whole = context(model.get(), ctx, gpu);
        auto repeat = context(model.get(), ctx, gpu);
        double repeat_seconds = 0;
        int64_t n = 0;
        std::vector<size_t> split_ends;
        std::vector<llama_token> continuation, free_continuation;
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
            const auto repeat_start = std::chrono::steady_clock::now();
            hybrid::prefill(repeat.get(), {tokens.begin() + n, tokens.begin() + n1}, n);
            repeat_seconds += seconds_since(repeat_start);
            const auto ends = boundaries(n1, 256, n);
            split_ends.insert(split_ends.end(), ends.begin(), ends.end());
            n = n1;
        }
        require(n == int64_t(tokens.size() - 1), "incomplete check chain");
        split_ends.push_back(tokens.size());
        hybrid::prefill(imported.get(), {tokens.back()}, n);
        hybrid::prefill(split.get(), {tokens.back()}, n);
        const auto repeat_start = std::chrono::steady_clock::now();
        hybrid::prefill(repeat.get(), {tokens.back()}, n);
        repeat_seconds += seconds_since(repeat_start);
        hybrid::prefill(whole.get(), tokens, 0);
        hybrid::prefill(free_whole.get(), tokens, 0);
        double kl_sum = 0, same_kl_sum = 0, same_diff = 0, repeat_kl_sum = 0, repeat_diff = 0;
        bool same_exact = true, repeat_exact = true, within_floor = true;
        auto positions = hybrid::json::array();
        int top1 = 0, same = 0, first_divergence = -1;
        for (int i = 0; i < k; ++i) {
            const auto token = hybrid::greedy(imported.get(), model.get());
            const auto local_logits = llama_get_logits_ith(split.get(), -1);
            const auto imported_logits = llama_get_logits_ith(imported.get(), -1);
            const auto repeat_logits = llama_get_logits_ith(repeat.get(), -1);
            const double diff = max_difference(imported_logits, local_logits, vocab);
            const double floor = max_difference(repeat_logits, local_logits, vocab);
            const bool exact = std::memcmp(imported_logits, local_logits, vocab * sizeof(float)) == 0;
            const bool local_exact = std::memcmp(repeat_logits, local_logits, vocab * sizeof(float)) == 0;
            const double same_kl = exact ? 0 : compare(imported_logits, local_logits, vocab);
            const double local_kl = local_exact ? 0 : compare(repeat_logits, local_logits, vocab);
            same_exact &= exact;
            repeat_exact &= local_exact;
            within_floor &= within_repeat_floor(exact, local_exact, same_kl, diff, local_kl, floor);
            same_kl_sum += same_kl;
            repeat_kl_sum += local_kl;
            same_diff = std::max(same_diff, diff);
            repeat_diff = std::max(repeat_diff, floor);
            positions.push_back({{"position", tokens.size() + i}, {"same_partition_kl_nats", same_kl},
                {"same_partition_max_abs_logit_diff", diff}, {"repeat_kl_nats", local_kl},
                {"repeat_max_abs_logit_diff", floor}, {"exact", exact}, {"repeat_exact", local_exact}});
            same += token == hybrid::greedy(split.get(), model.get());
            top1 += token == hybrid::greedy(whole.get(), model.get());
            const auto free_token = hybrid::greedy(free_whole.get(), model.get());
            continuation.push_back(token);
            free_continuation.push_back(free_token);
            if (first_divergence < 0 && token != free_token) { first_divergence = i; }
            kl_sum += compare(llama_get_logits_ith(imported.get(), -1), llama_get_logits_ith(whole.get(), -1), vocab);
            // Teacher forcing uses the imported continuation for both numerical references.
            hybrid::prefill(imported.get(), {token}, tokens.size() + i);
            hybrid::prefill(split.get(), {token}, tokens.size() + i);
            const auto step_start = std::chrono::steady_clock::now();
            hybrid::prefill(repeat.get(), {token}, tokens.size() + i);
            repeat_seconds += seconds_since(step_start);
            hybrid::prefill(whole.get(), {token}, tokens.size() + i);
            hybrid::prefill(free_whole.get(), {free_token}, tokens.size() + i);
        }
        const double kl = kl_sum / k, agreement = double(top1) / k;
        hybrid::json result = {{"stage", "equivalence"}, {"depth", tokens.size()}, {"k", k},
            {"same_partition_agreed", same}, {"top1_agreed", top1}, {"top1_agreement", agreement},
            {"mean_kl_nats", kl}, {"free_running_first_divergence", first_divergence},
            {"kl_limit", .01}, {"top1_minimum", .95}, {"same_partition_exact", same_exact}, {"same_partition_within_floor", within_floor},
            {"same_partition_kl_nats", same_kl_sum / k}, {"same_partition_max_abs_logit_diff", same_diff},
            {"same_partition_repeat_exact", repeat_exact}, {"same_partition_repeat_kl_nats", repeat_kl_sum / k},
            {"same_partition_repeat_max_abs_logit_diff", repeat_diff}, {"same_partition_positions", positions},
            {"same_partition_repeat_seconds", repeat_seconds}, {"backend", backend},
            {"numerical_within_bounds", kl <= .01 && agreement >= .95}};
        // Run noise partitions after releasing the comparison contexts.
        imported.reset();
        split.reset();
        whole.reset();
        free_whole.reset();
        repeat.reset();
        const auto noise_start = std::chrono::steady_clock::now();
        const auto whole_ends = boundaries(tokens.size(), 256);
        std::vector<std::vector<size_t>> noise_ends;
        auto partitions = hybrid::json::array();
        double noise_max = 0;
        int noise_top1_min = k, noise_divergence = -1;
        for (int requested : chunks) {
            size_t chunk = std::min<size_t>(requested, tokens.size() - 1);
            auto ends = boundaries(tokens.size(), chunk);
            while (ends[ends.size() - 2] == split_ends[split_ends.size() - 2] || ends == whole_ends || ends == split_ends || std::find(noise_ends.begin(), noise_ends.end(), ends) != noise_ends.end()) {
                if (--chunk == 0) { break; }
                ends = boundaries(tokens.size(), chunk);
            }
            if (chunk == 0) { continue; }
            noise_ends.push_back(ends);
            auto reference = context(model.get(), ctx, gpu);
            auto alternate = context(model.get(), ctx, gpu);
            auto free_alternate = context(model.get(), ctx, gpu);
            hybrid::prefill(reference.get(), tokens, 0);
            prefill_partition(alternate.get(), tokens, chunk);
            const auto first_free_token = hybrid::greedy(alternate.get(), model.get());
            hybrid::load_state(free_alternate.get(), hybrid::save_state(alternate.get()));
            double noise_sum = 0;
            int noise_top1 = 0, divergence = -1;
            for (int i = 0; i < k; ++i) {
                noise_sum += compare(llama_get_logits_ith(alternate.get(), -1), llama_get_logits_ith(reference.get(), -1), vocab);
                noise_top1 += hybrid::greedy(alternate.get(), model.get()) == hybrid::greedy(reference.get(), model.get());
                const auto free_token = i == 0 ? first_free_token : hybrid::greedy(free_alternate.get(), model.get());
                if (divergence < 0 && free_token != free_continuation[i]) { divergence = i; }
                hybrid::prefill(reference.get(), {continuation[i]}, tokens.size() + i);
                hybrid::prefill(alternate.get(), {continuation[i]}, tokens.size() + i);
                hybrid::prefill(free_alternate.get(), {free_token}, tokens.size() + i);
            }
            const double noise_kl = noise_sum / k;
            require(std::isfinite(noise_kl), "nonfinite noise KL");
            noise_max = std::max(noise_max, noise_kl);
            noise_top1_min = std::min(noise_top1_min, noise_top1);
            if (divergence >= 0 && (noise_divergence < 0 || divergence < noise_divergence)) { noise_divergence = divergence; }
            partitions.push_back({{"requested_chunk_tokens", requested}, {"chunk_tokens", chunk}, {"batches", ends.size()}, {"last_prefill_boundary", ends[ends.size() - 2]},
                {"noise_kl_nats", noise_kl}, {"noise_top1_agreed", noise_top1}, {"noise_top1_agreement", double(noise_top1) / k},
                {"noise_free_running_first_divergence", divergence}});
        }
        require(!partitions.empty(), "no distinct local noise partition");
        result["noise_partitions"] = partitions;
        result["noise_reference_seconds"] = seconds_since(noise_start);
        result["noise_reference_variant"] = "alternate_state_snapshot";
        result["hybrid_last_prefill_boundary"] = split_ends[split_ends.size() - 2];
        result["noise_kl_nats"] = result["noise_kl_nats_max"] = noise_max;
        result["noise_top1_agreed"] = noise_top1_min;
        result["noise_top1_agreement"] = double(noise_top1_min) / k;
        result["noise_free_running_first_divergence"] = noise_divergence;
        result["kl_to_noise_ratio"] = noise_max > 0 ? hybrid::json(kl / noise_max) : hybrid::json(nullptr);
        result["kl_to_noise_ratio_status"] = noise_max > 0 ? "finite" : kl == 0 ? "both_zero" : "zero_noise_floor";
        const bool uncalibrated = fixture.value("uncalibrated", false);
        const bool passed = same == k && within_floor && (gpu || result["numerical_within_bounds"].get<bool>());
        result["status_rule"] = uncalibrated ? "uncalibrated_client_noise" : repeat_exact ? "same_backend_same_partition_exact" : "same_backend_same_partition_measured_floor";
        result["status"] = uncalibrated ? "uncalibrated" : passed ? "passed" : "failed";
        result["exit_code"] = uncalibrated ? 5 : passed ? 0 : 3;
        std::ofstream target(argv[4]);
        target << result.dump(2) << '\n';
        require(target.good(), "cannot write check result");
        std::cout << result.dump() << '\n';
        return result["exit_code"].get<int>();
    } catch (const std::exception & error) {
        std::cerr << "check failed: " << error.what() << '\n';
        return 2;
    }
}
