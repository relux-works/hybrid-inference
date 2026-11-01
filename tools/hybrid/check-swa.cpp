#include "hybrid-state.h"
#include "llama-io.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"

#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

class writer : public llama_io_write_i {
public:
    std::string data;
    void write(const void * src, size_t n) override { data.append((const char *) src, n); }
    template <typename T> void scalar(T value) { write(&value, sizeof(value)); }
    void write_tensor(ggml_tensor * t, size_t at, size_t n) override {
        const size_t offset = data.size();
        data.resize(offset + n);
        ggml_backend_tensor_get(t, &data[offset], at, n);
    }
    size_t n_bytes() override { return data.size(); }
};

class reader : public llama_io_read_i {
public:
    reader(const std::string & data, bool validate = false, int fail_tensor = -1) :
        data(data), validate(validate), fail_tensor(fail_tensor) {}
    ~reader() { commit(); }
    void read(void * dst, size_t n) override {
        require(n <= data.size() - offset, "truncated synthetic state");
        memcpy(dst, data.data() + offset, n);
        offset += n;
    }
    void read_tensor(ggml_tensor * t, size_t at, size_t n) override {
        require(n <= data.size() - offset, "truncated synthetic tensor");
        if (fail_tensor == calls++) { throw std::runtime_error("injected tensor read failure"); }
        if (!validate) { pending.push_back({t, at, n, offset}); }
        offset += n;
    }
    void discard() override { pending.clear(); }
    void commit() {
        for (const auto & p : pending) { ggml_backend_tensor_set(p.t, data.data() + p.offset, p.at, p.n); }
        pending.clear();
    }
    size_t n_bytes() override { return offset; }
private:
    struct write { ggml_tensor * t; size_t at, n, offset; };
    const std::string & data;
    const bool validate;
    const int fail_tensor;
    int calls = 0;
    size_t offset = 0;
    std::vector<write> pending;
};

static std::string cache_record(llama_kv_cache * cache, const llama_hparams & hp, const std::vector<int> & positions, uint32_t streams = 1) {
    writer out;
    out.scalar<uint32_t>(streams);
    out.scalar<uint32_t>(positions.size());
    for (int pos : positions) {
        out.scalar<int32_t>(pos);
        out.scalar<uint32_t>(1);
        out.scalar<int32_t>(0);
    }
    if (positions.empty()) { return out.data; }
    const auto layers = cache->get_layer_ids();
    out.scalar<uint32_t>(0);
    out.scalar<uint32_t>(layers.size());
    out.scalar<uint32_t>(0);
    out.scalar<uint32_t>(0);
    for (bool value : {false, true}) {
        for (auto il : layers) {
            const size_t row = ggml_row_size(GGML_TYPE_Q8_0, value ? hp.n_embd_v_gqa(il) : hp.n_embd_k_gqa(il));
            out.scalar<int32_t>(GGML_TYPE_Q8_0);
            out.scalar<uint64_t>(row);
            for (int pos : positions) {
                out.data.append(row, char(1 + (pos + il + value) % 100));
            }
        }
    }
    for (uint32_t i = 1; i < streams; ++i) { out.scalar<uint32_t>(0); }
    return out.data;
}

static std::string snapshot(llama_kv_cache_iswa & cache, int seq = -1) {
    writer out;
    cache.state_write(out, seq, 0);
    return out.data;
}

static void run(llm_arch arch, bool unified) {
    hybrid::model_ptr model(llama_model_create(arch, llama_model_default_params()));
    auto & hp = model->hparams;
    hp = {};
    hp.n_layer_all = arch == LLM_ARCH_GEMMA4 ? 8 : 4;
    hp.n_embd = 64;
    hp.n_ctx_train = 512;
    hp.n_head_arr.fill(1);
    hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 64;
    hp.n_embd_head_k_swa = hp.n_embd_head_v_swa = 32;
    hp.n_swa = 4;
    hp.swa_type = LLAMA_SWA_TYPE_STANDARD;
    std::iota(hp.kv_source_impl.begin(), hp.kv_source_impl.end(), 0);
    hp.set_swa_pattern(arch == LLM_ARCH_GEMMA4 ? 3 : 4);
    if (arch == LLM_ARCH_GEMMA4) {
        hp.n_layer_kv_from_start = 6;
        hp.is_swa_impl[7] = false;
    }
    // Admission accepts the requested architectures and refuses a different dense architecture.
    model->arch = LLM_ARCH_LLAMA;
    bool refused = false;
    try { hybrid::model_layout(model.get()); }
    catch (const std::runtime_error & error) { refused = std::string(error.what()) == "unsupported hybrid architecture"; }
    require(refused, "unknown native architecture accepted");
    model->arch = arch;
    // Server import admission uses the same gate, including for otherwise valid standard-SWA models.
    for (auto admitted : {LLM_ARCH_QWEN35, LLM_ARCH_GEMMA4, LLM_ARCH_MUSE_GLIMMER}) {
        model->arch = admitted;
        hybrid::require_supported_architecture(model.get());
    }
    for (auto denied : {LLM_ARCH_UNKNOWN, LLM_ARCH_LLAMA, LLM_ARCH_GEMMA2, LLM_ARCH_GEMMA3, LLM_ARCH_OPENAI_MOE, LLM_ARCH_COHERE2, LLM_ARCH_QWEN35MOE}) {
        model->arch = denied;
        refused = false;
        try { hybrid::require_supported_architecture(model.get()); }
        catch (const std::runtime_error & error) { refused = std::string(error.what()) == "unsupported hybrid architecture"; }
        require(refused, "server_import_architecture_admission: unsupported architecture accepted");
    }
    model->arch = arch;
    std::cout << "PASS server_import_architecture_admission\n";
    const auto layout = hybrid::model_layout(model.get());
    std::cout << "LAYOUT " << hybrid::json({{"layout_fingerprint", hybrid::layout_fingerprint(layout)}, {"model_layout", layout}}).dump() << '\n';
    llama_kv_cache_iswa cache(*model, false, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                             false, false, false, unified, 512, 2, 256, nullptr, nullptr,
                             [&](int32_t il) { return hp.has_kv(il) ? -1 : 6 - (hp.is_swa(il) ? 2 : 1); }, nullptr);
    auto record = [&](const std::vector<int> & base, const std::vector<int> & window) {
        return cache_record(cache.get_base(), hp, base) + cache_record(cache.get_swa(), hp, window);
    };
    const auto prefix = record({0, 1, 2, 3, 4, 5, 6, 7}, {6, 7, 4, 5});
    for (int seq : {0, 1}) {
        reader input(prefix);
        cache.state_read(input, seq, 0);
    }
    const auto before = snapshot(cache);
    const auto neighbor = snapshot(cache, 1);
    const auto good = record({8, 9, 10}, {9, 10, 7, 8});
    auto reject = [&](const char * name, const std::string & bytes, int end, bool validate, int fail_tensor = -1) {
        bool rejected = false;
        {
            reader input(bytes, validate, fail_tensor);
            try { cache.state_read_append(input, 0, 8, end, validate); }
            catch (const std::runtime_error &) { input.discard(); rejected = true; }
        }
        require(rejected, name);
        if (snapshot(cache) != before) {
            throw std::runtime_error(std::string(name) + ": rejected import changed cache bytes or cells");
        }
        require(snapshot(cache, 1) == neighbor, "rejected import changed neighbor");
        std::cout << "PASS " << name << '\n';
    };
    // Each malformed position set or short buffer must fail before changing either cache.
    reject("window_endpoint_mismatch", record({8, 9, 10}, {6, 7, 8, 9}), 11, true);
    reject("window_duplicate_position", record({8, 9, 10}, {7, 8, 9, 9}), 11, true);
    reject("window_missing_position", record({8, 9, 10}, {8, 9, 10}), 11, true);
    reject("window_empty", record({8, 9, 10}, {}), 11, true);
    reject("full_endpoint_mismatch", good, 12, true);
    reject("window_truncated_header", cache_record(cache.get_base(), hp, {8, 9, 10}) + std::string(3, '\0'), 11, true);
    reject("window_truncated_tensor", good.substr(0, good.size() - 1), 11, true);
    const size_t window_at = cache_record(cache.get_base(), hp, {8, 9, 10}).size();
    auto changed32 = [&](size_t at, int32_t value) {
        auto changed = good;
        memcpy(&changed[at], &value, sizeof(value));
        return changed;
    };
    reject("window_invalid_source_sequence", changed32(window_at + 16, -1), 11, true);
    reject("window_layer_count_mismatch", changed32(window_at + 8 + 4 * 12 + 4, 999), 11, true);
    reject("window_key_type_mismatch", changed32(window_at + 8 + 4 * 12 + 16, GGML_TYPE_F32), 11, true);
    // Inject after all full-cache writes are queued and window metadata is replaced; discard must preserve physical state.
    reject("failed_apply_restores_both_caches_exactly", good, 11, false, 2 * cache.get_base()->get_layer_ids().size() + 1);
    reject("failed_apply_truncation_restores_exactly", good.substr(0, good.size() - 1), 11, false);
    // A post-check failure after both caches change must discard tensor writes and restore both sequences.
    for (bool bad_length : {false, true}) {
        const char * name = bad_length ? "postcheck_length_rollback" : "postcheck_endpoint_rollback";
        const auto bytes = good + (bad_length ? "x" : "");
        bool checked = false;
        bool rejected = false;
        {
            reader input(bytes);
            try {
                cache.state_read_append(input, 0, 8, 11, false, [&] {
                    checked = true;
                    require(cache.get_base()->seq_pos_max(0) == 10 && cache.get_swa()->seq_pos_min(0) == 7,
                            "post-check did not run after both cache updates");
                    if (input.n_bytes() != bytes.size() || cache.seq_pos_max(0) != (bad_length ? 10 : 11)) {
                        throw std::runtime_error("append length does not match the declared range");
                    }
                });
            } catch (const std::runtime_error & error) {
                rejected = std::string(error.what()) == "append length does not match the declared range";
            }
        }
        require(checked && rejected, "post-check did not reject the injected mismatch");
        require(snapshot(cache) == before && snapshot(cache, 1) == neighbor, name);
        std::cout << "PASS " << name << '\n';
    }
    // Validation is read-only; the positive append replaces the complete window, including positions before p0.
    {
        reader input(good, true);
        bool checked = false;
        cache.state_read_append(input, 0, 8, 11, true, [&] { checked = true; });
        require(!checked, "validation invoked the apply post-check");
        require(input.n_bytes() == good.size(), "validation did not consume both caches");
    }
    require(snapshot(cache) == before, "validation changed state");
    {
        reader input(good);
        bool checked = false;
        cache.state_read_append(input, 0, 8, 11, false, [&] {
            checked = true;
            require(input.n_bytes() == good.size() && cache.seq_pos_max(0) == 10, "valid post-check failed");
        });
        require(checked, "valid append skipped post-check");
    }
    require(cache.get_base()->seq_pos_min(0) == 0 && cache.get_base()->seq_pos_max(0) == 10, "full prefix lost");
    require(cache.get_swa()->seq_pos_min(0) == 7 && cache.get_swa()->seq_pos_max(0) == 10, "wrong window");
    require(snapshot(cache, 1) == neighbor, "positive append changed neighbor");
    writer delta;
    cache.state_write_range(delta, 0, 8, 11, 0);
    const auto exported = cache_record(cache.get_base(), hp, {8, 9, 10}, cache.get_base()->get_n_stream()) +
                          cache_record(cache.get_swa(), hp, {9, 10, 7, 8}, cache.get_swa()->get_n_stream());
    require(delta.data == exported, "export did not preserve full range plus wrapped whole window");
    std::cout << "PASS full_append_whole_window_replace_export " << model->arch_name() << " unified=" << unified << '\n';
    // An initial import must include a shorter-than-window prefix and both allocated cache types.
    llama_kv_cache_iswa fresh(*model, false, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                             false, false, false, unified, 512, 2, 256, nullptr, nullptr, nullptr, nullptr);
    const auto initial = record({0, 1}, {0, 1});
    {
        reader input(initial, true);
        fresh.state_read_append(input, 0, 0, 2, true);
    }
    {
        reader input(initial);
        fresh.state_read_append(input, 0, 0, 2, false);
    }
    require(fresh.seq_pos_max(0) == 1, "initial import failed");
    std::cout << "PASS initial_short_window_import\n";
}

int main() {
    try {
        llama_backend_init();
        for (auto arch : {LLM_ARCH_GEMMA4, LLM_ARCH_MUSE_GLIMMER}) {
            for (bool unified : {false, true}) { run(arch, unified); }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
