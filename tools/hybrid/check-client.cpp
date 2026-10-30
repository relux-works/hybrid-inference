#include "hybrid-state.h"
#include "llama-ext.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static std::string read(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open fixture");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static void write(const std::string & path, const std::string & data) {
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), data.size());
    require(file.good(), "cannot write fixture");
}

static size_t rss(const std::string & name) {
    std::istringstream lines(read("/proc/self/status"));
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(name + ":") == 0) {
            return std::stoull(line.substr(name.size() + 1)) * 1024;
        }
    }
    throw std::runtime_error("missing RSS counter");
}

static std::string state(llama_context * ctx, int seq) {
    std::string out(llama_state_seq_get_size(ctx, seq), '\0');
    require(llama_state_seq_get_data(ctx, (uint8_t *) &out[0], out.size(), seq) == out.size(), "save failed");
    return out;
}

static size_t append(llama_context * ctx, const std::string & raw, int n0, int n1) {
    return llama_state_seq_set_data_append(ctx, (const uint8_t *) raw.data(), raw.size(), 0, n0, n1);
}

static void prepare(llama_model * model, const std::string & golden, const std::string & output) {
    // Fixed 128-token suffix and context size isolate append cost from prefix depth and allocation size.
    for (int depth : {512, 2048, 8192}) {
        const auto stem = golden + "/prompt-" + std::to_string(depth);
        const auto metadata = hybrid::parse_json(read(stem + ".tokens.json"));
        const auto tokens = metadata.at("tokens").get<std::vector<llama_token>>();
        const auto full = hybrid::read_delta(read(stem + ".full.delta"), metadata.at("model_sha256"), 0);
        auto ctx = hybrid::make_context(model, 9216);
        hybrid::load_state(ctx.get(), full.state);
        hybrid::prefill(ctx.get(), {tokens.back()}, depth - 1);
        const auto target = output + "/cost-" + std::to_string(depth);
        write(target + ".prefix.state", hybrid::save_state(ctx.get()));
        hybrid::prefill(ctx.get(), {tokens.end() - 128, tokens.end()}, depth);
        write(target + ".range.state", hybrid::export_range(ctx.get(), depth, depth + 128));
        write(target + ".expected.state", hybrid::save_state(ctx.get()));
        std::cout << "PASS prepared cost fixture n0=" << depth << " suffix=128\n";
    }
}

static void measure(llama_model * model, const std::string & stem, int n0) {
    auto ctx = hybrid::make_context(model, 9216);
    size_t prefix_bytes;
    {
        const auto prefix = read(stem + ".prefix.state");
        prefix_bytes = prefix.size();
        hybrid::load_state(ctx.get(), prefix);
    }
    const auto raw = read(stem + ".range.state");
    const size_t recurrent_bytes = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    // Reset only this process's peak counter after setup. Equality checks run outside the measured interval.
    write("/proc/self/clear_refs", "5\n");
    const auto before = rss("VmRSS");
    const auto start = std::chrono::steady_clock::now();
    const auto result = append(ctx.get(), raw, n0, n0 + 128);
    const auto end = std::chrono::steady_clock::now();
    const auto peak = rss("VmHWM");
    require(result == raw.size(), "measured append failed");
    std::cout << "COST " << hybrid::json{{"n0", n0}, {"suffix", 128}, {"prefix_bytes", prefix_bytes},
        {"recurrent_backup_bytes", recurrent_bytes}, {"milliseconds", std::chrono::duration<double, std::milli>(end - start).count()},
        {"rss_before", before}, {"peak_rss", peak}, {"extra_peak_bytes", peak > before ? peak - before : 0}}.dump() << '\n';
    require(hybrid::save_state(ctx.get()) == read(stem + ".expected.state"), "measured append differs from source");
    std::cout << "PASS measured append equals source n0=" << n0 << '\n';
}

static void capacity(llama_model * model, const std::string & stem) {
    const auto meta = hybrid::parse_json(read(stem + ".tokens.json"));
    const int n0 = meta.at("n0"), n1 = meta.at("n1");
    const auto delta = hybrid::read_delta(read(stem + ".delta"), meta.at("model_sha256"), n0);
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_seq_max = 2; cp.kv_unified = true;
    cp.n_batch = 256; cp.n_ubatch = 256; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = false; cp.op_offload = false;
    hybrid::context_ptr ctx(llama_init_from_model(model, cp));
    require(bool(ctx), "capacity context failed");
    hybrid::load_state(ctx.get(), read(stem + ".prefix.state"));
    auto batch = llama_batch_init(2, 0, 1);
    for (int i = 0; i < 2; ++i) {
        batch.token[i] = 1; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 1; batch.logits[i] = false;
    }
    batch.n_tokens = 2;
    const int decoded = llama_decode(ctx.get(), batch);
    llama_batch_free(batch);
    require(decoded == 0, "neighbor prefill failed");
    const auto before = state(ctx.get(), 0), neighbor = state(ctx.get(), 1);
    // The valid 128-cell record fits the cache shape but only 127 cells are free. Allocation fails after validation.
    require(n0 == 383 && n1 == 511, "wrong capacity fixture");
    require(append(ctx.get(), delta.state, n0, n1) == 0, "capacity failure accepted");
    require(state(ctx.get(), 0) == before, "capacity rejection changed prefix KV or recurrent bytes");
    require(state(ctx.get(), 1) == neighbor, "capacity rejection changed neighbor bytes");
    std::cout << "PASS test_append_capacity_failure_preserves_prefix_and_neighbor\n";
    require(llama_memory_seq_rm(llama_get_memory(ctx.get()), 1, -1, -1), "neighbor removal failed");
    require(append(ctx.get(), delta.state, n0, n1) == delta.state.size(), "capacity positive control failed");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == n1 - 1, "positive endpoint mismatch");
    std::cout << "PASS capacity positive control after freeing neighbor\n";
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 4, "usage: llama-hybrid-check-client MODEL --prepare GOLDEN_DIR OUTPUT_DIR | --measure STEM N0 | --capacity GOLDEN_STEM");
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "model load failed");
        const std::string mode = argv[2];
        if (mode == "--prepare" && argc == 5) { prepare(model.get(), argv[3], argv[4]); }
        else if (mode == "--measure" && argc == 5) { measure(model.get(), argv[3], std::stoi(argv[4])); }
        else if (mode == "--capacity" && argc == 4) { capacity(model.get(), argv[3]); }
        else { throw std::runtime_error("invalid mode or arguments"); }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
