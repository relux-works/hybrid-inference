#include "hybrid-state.h"
#include "llama-io.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "gguf.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <cstdio>
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

static std::string window_record(llama_kv_cache * cache, const llama_hparams & hp, const std::vector<int> & positions, uint32_t streams = 1) {
    const auto legacy = cache_record(cache, hp, positions, streams);
    writer out;
    out.scalar<uint32_t>(streams | 0xc0000000u);
    out.scalar<uint32_t>(positions.size());
    if (positions.empty()) { return out.data; }
    out.scalar<uint32_t>(cache->get_size());
    out.scalar<uint32_t>(positions.size());
    for (uint32_t i = 0; i < positions.size(); ++i) { out.scalar<uint32_t>(i); }
    return out.data + legacy.substr(8);
}

static std::string snapshot(llama_kv_cache_iswa & cache, int seq = -1) {
    writer out;
    cache.get_base()->state_write(out, seq, 0);
    cache.get_swa()->state_write(out, seq, 0);
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
        return cache_record(cache.get_base(), hp, base) + window_record(cache.get_swa(), hp, window);
    };
    const auto prefix = cache_record(cache.get_base(), hp, {0, 1, 2, 3, 4, 5, 6, 7}) +
                        cache_record(cache.get_swa(), hp, {6, 7, 4, 5});
    for (int seq : {0, 1}) {
        reader input(prefix);
        cache.state_read(input, seq, 0);
    }
    const auto before = snapshot(cache);
    const auto neighbor = snapshot(cache, 1);
    const auto good = record({8, 9, 10}, {9, 10, 7, 8});
    auto reject = [&](const char * name, const std::string & bytes, int end, bool validate, int fail_tensor = -1, const char * expected = nullptr) {
        bool rejected = false;
        {
            reader input(bytes, validate, fail_tensor);
            try { cache.state_read_append(input, 0, 8, end, validate); }
            catch (const std::runtime_error & error) {
                input.discard();
                require(!expected || error.what() == std::string(expected), "wrong layout rejection");
                rejected = true;
            }
        }
        require(rejected, name);
        if (snapshot(cache) != before) {
            throw std::runtime_error(std::string(name) + ": rejected import changed cache bytes or cells");
        }
        require(snapshot(cache, 1) == neighbor, "rejected import changed neighbor");
        std::cout << "PASS " << name << '\n';
    };
    // Both compacted v0 and full-payload v1 transfers refuse without changing either sequence.
    const auto legacy = cache_record(cache.get_base(), hp, {8, 9, 10}) + cache_record(cache.get_swa(), hp, {9, 10, 7, 8});
    auto previous = good;
    const uint32_t previous_version = 0x80000001u;
    const size_t window_offset = cache_record(cache.get_base(), hp, {8, 9, 10}).size();
    memcpy(&previous[window_offset], &previous_version, sizeof(previous_version));
    for (bool validate : {false, true}) {
        reject("swa_legacy_compacted_record_refused", legacy, 11, validate, -1, "unsupported sliding-window record version");
        reject("swa_previous_layout_version_refused", previous, 11, validate, -1, "unsupported sliding-window record version");
    }
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
    reject("window_invalid_source_sequence", changed32(window_at + 40, -1), 11, true);
    reject("window_layer_count_mismatch", changed32(window_at + 32 + 4 * 12 + 4, 999), 11, true);
    reject("window_key_type_mismatch", changed32(window_at + 32 + 4 * 12 + 16, GGML_TYPE_F32), 11, true);
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
    const auto window = cache_record(cache.get_swa(), hp, {9, 10, 7, 8}, cache.get_swa()->get_n_stream());
    writer physical;
    physical.scalar<uint32_t>(cache.get_swa()->get_n_stream() | 0xc0000000u);
    physical.scalar<uint32_t>(4);
    physical.scalar<uint32_t>(cache.get_swa()->get_size());
    physical.scalar<uint32_t>(4);
    for (uint32_t i = 0; i < 4; ++i) { physical.scalar<uint32_t>(i); }
    const auto exported = cache_record(cache.get_base(), hp, {8, 9, 10}, cache.get_base()->get_n_stream()) +
                          physical.data + window.substr(8);
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
    { reader input(prefix); cache.state_read(input, 0, 0); }
    // Physical layout bounds, uniqueness and completeness are checked before any tensor write.
    const size_t layout_at = cache_record(cache.get_base(), hp, {8, 9, 10}, cache.get_base()->get_n_stream()).size();
    auto bad_layout = [&](const char * name, size_t offset, uint32_t value, const char * expected) {
        auto bytes = exported;
        memcpy(&bytes[layout_at + offset], &value, sizeof(value));
        reject(name, bytes, 11, true, -1, expected);
        reject(name, bytes, 11, false, -1, expected);
    };
    bad_layout("swa_layout_zero_capacity", 8, 0, "invalid sliding-window layout size");
    bad_layout("swa_layout_head_out_of_bounds", 12, cache.get_swa()->get_size() + 1, "invalid sliding-window layout size");
    bad_layout("swa_layout_index_out_of_bounds", 16, cache.get_swa()->get_size(), "invalid sliding-window cell index");
    bad_layout("swa_layout_duplicate_index", 20, 0, "invalid sliding-window cell index");
    bad_layout("swa_layout_incomplete_window", 4, 3, "incomplete sliding-window layout");
    bad_layout("swa_layout_negative_position", 32, UINT32_MAX, "invalid sliding-window position");
    bad_layout("swa_layout_position_out_of_range", 32, 11, "cache range position mismatch");
    bad_layout("swa_layout_duplicate_position", 32, 10, "cache range position mismatch");
    // A failure after restoring the physical allocator head rolls back cells, head and deferred tensor writes.
    writer saved;
    cache.state_write(saved, 0, 0);
    auto different_head = exported;
    const uint32_t head = cache.get_swa()->get_size() - 1;
    memcpy(&different_head[layout_at + 12], &head, sizeof(head));
    bool rejected = false;
    {
        reader input(different_head);
        try {
            cache.state_read_append(input, 0, 8, 11, false, [&] {
                writer applied;
                cache.get_swa()->state_write(applied, 0, LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT);
                uint32_t observed;
                memcpy(&observed, applied.data.data() + 12, sizeof(observed));
                require(observed == head, "physical head was not applied");
                throw std::runtime_error("physical post-check");
            });
        } catch (const std::runtime_error & error) {
            rejected = std::string(error.what()) == "physical post-check";
        }
    }
    writer restored;
    cache.state_write(restored, 0, 0);
    require(rejected && saved.data == restored.data && snapshot(cache, 1) == neighbor, "swa_layout_postcheck_rollback");
    { reader input(exported); cache.state_read_append(input, 0, 8, 11, false); }
    require(snapshot(cache, 1) == neighbor, "physical import changed neighbor");
    std::cout << "PASS swa_physical_layout_roundtrip_and_refusals\n";
    // Ordinary sequence and PARTIAL_ONLY records retain the independent legacy byte layout.
    writer ordinary;
    cache.state_write(ordinary, 0, 0);
    const auto legacy_window = cache_record(cache.get_swa(), hp, {9, 10, 7, 8}, cache.get_swa()->get_n_stream());
    const auto legacy_sequence = cache_record(cache.get_base(), hp, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, cache.get_base()->get_n_stream()) + legacy_window;
    require(ordinary.data == legacy_sequence, "swa_generic_state_legacy_bytes");
    writer partial, legacy_partial;
    cache.state_write(partial, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    cache.get_swa()->state_write(legacy_partial, 0, 0);
    require(partial.data == legacy_window && partial.data == legacy_partial.data, "swa_checkpoint_legacy_bytes");
    std::cout << "PASS swa_generic_and_checkpoint_legacy_bytes\n";

    // A partially freed ring has indices [7,11), so prefix placement cannot pass this round trip.
    cache.clear(true);
    fresh.clear(true);
    std::vector<int> positions(11);
    std::iota(positions.begin(), positions.end(), 0);
    const auto nonprefix = cache_record(cache.get_base(), hp, positions) + cache_record(cache.get_swa(), hp, positions);
    { reader input(nonprefix); cache.state_read(input, 0, 0); }
    cache.get_swa()->seq_rm(0, 0, 7);
    writer nonprefix_delta, source_layout, imported_layout;
    cache.state_write_range(nonprefix_delta, 0, 0, 11, 0);
    { reader input(nonprefix_delta.data); fresh.state_read_append(input, 0, 0, 11, false); }
    cache.get_swa()->state_write(source_layout, 0, LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT);
    fresh.get_swa()->state_write(imported_layout, 0, LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT);
    require(source_layout.data == imported_layout.data, "swa_nonprefix_physical_indices");
    require(fresh.get_swa()->get_cells(0).seq_has(7, 0) && !fresh.get_swa()->get_cells(0).seq_has(0, 0),
            "swa_nonprefix_physical_indices");
    std::cout << "PASS swa_nonprefix_physical_indices\n";

    // Hundreds of masked rows add metadata but no tensor bytes, with the retained four rows unchanged.
    cache.clear(true);
    positions.resize(273);
    std::iota(positions.begin(), positions.end(), 0);
    const auto occupied = cache_record(cache.get_base(), hp, positions) + cache_record(cache.get_swa(), hp, positions);
    { reader input(occupied); cache.state_read(input, 0, 0); }
    writer sparse;
    cache.get_swa()->state_write(sparse, 0, LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT);
    const size_t layers = cache.get_swa()->get_layer_ids().size();
    size_t row_bytes = 0;
    for (auto il : cache.get_swa()->get_layer_ids()) {
        row_bytes += ggml_row_size(GGML_TYPE_Q8_0, hp.n_embd_k_gqa(il)) + ggml_row_size(GGML_TYPE_Q8_0, hp.n_embd_v_gqa(il));
    }
    const size_t expected_bytes = 32 + 24 * layers + 16 * positions.size() + 4 * row_bytes +
                                  4 * (cache.get_swa()->get_n_stream() - 1);
    require(sparse.data.size() == expected_bytes, "swa_masked_rows_metadata_only");
    writer compact;
    cache.get_swa()->state_write(compact, 0, 0);
    require(sparse.data.size() - compact.data.size() == 8 + 16 * positions.size() - 4 * 12,
            "swa_masked_rows_metadata_only");
    std::cout << "PASS swa_masked_rows_metadata_only bytes=" << sparse.data.size() << '\n';
}

static void tiny_model(const char * path) {
    gguf_context * file = gguf_init_empty();
    ggml_context * weights = ggml_init({2 * 1024 * 1024, nullptr, false});
    require(file && weights, "cannot allocate tiny model");
    gguf_set_val_str(file, "general.architecture", "muse-glimmer");
    gguf_set_val_str(file, "tokenizer.ggml.model", "llama");
    const char * tokens[] = {"<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"};
    const int32_t types[] = {2, 3, 3, 1, 1, 1, 1, 1};
    gguf_set_arr_str(file, "tokenizer.ggml.tokens", tokens, 8);
    gguf_set_arr_data(file, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 8);
    gguf_set_val_u32(file, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(file, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(file, "tokenizer.ggml.eos_token_id", 2);
    for (const auto & item : std::vector<std::pair<const char *, uint32_t>> {
            {"context_length", 1024}, {"embedding_length", 64}, {"block_count", 4},
            {"feed_forward_length", 64}, {"attention.head_count", 1}, {"attention.head_count_kv", 1},
            {"attention.sliding_window", 4}, {"rope.dimension_count", 64}}) {
        gguf_set_val_u32(file, (std::string("muse-glimmer.") + item.first).c_str(), item.second);
    }
    gguf_set_val_f32(file, "muse-glimmer.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_f32(file, "muse-glimmer.logit_scale", 1.0f);
    uint32_t random = 42;
    auto tensor = [&](const std::string & name, int cols, int rows = 0) {
        auto * value = rows ? ggml_new_tensor_2d(weights, GGML_TYPE_F32, cols, rows) :
                             ggml_new_tensor_1d(weights, GGML_TYPE_F32, cols);
        ggml_set_name(value, name.c_str());
        for (int64_t i = 0; i < ggml_nelements(value); ++i) {
            random = random * 1664525u + 1013904223u;
            ((float *) value->data)[i] = rows ? (int(random >> 16) - 32768) / 262144.0f : 1.0f;
        }
        gguf_add_tensor(file, value);
    };
    tensor("token_embd.weight", 64, 8);
    tensor("output.weight", 64, 8);
    tensor("output_norm.weight", 64);
    for (int il = 0; il < 4; ++il) {
        const auto prefix = "blk." + std::to_string(il) + ".";
        for (const char * name : {"attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "ffn_norm", "post_ffw_norm"}) {
            tensor(prefix + name + ".weight", 64);
        }
        for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "ffn_gate", "ffn_down", "ffn_up"}) {
            tensor(prefix + name + ".weight", 64, 64);
        }
    }
    FILE * output = fopen(path, "wbx");
    require(output, "tiny model output exists or cannot be created");
    const bool ok = gguf_write_to_file_ptr(file, output, false);
    const int close_result = fclose(output);
    gguf_free(file);
    ggml_free(weights);
    require(ok && close_result == 0, "cannot write tiny model");
}

static void cells(llama_context * ctx, const char * name, int end) {
    auto * memory = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
    require(memory, "tiny model did not allocate SWA cache");
    auto * cache = memory->get_swa();
    const auto & slots = cache->get_cells(0);
    auto rows = hybrid::json::array();
    for (uint32_t i = 0; i < slots.size(); ++i) {
        if (slots.is_empty(i) || !slots.seq_has(i, 0) || slots.pos_get(i) < end - 4) { continue; }
        rows.push_back({{"cell", i}, {"position", slots.pos_get(i)}, {"sequences", slots.seq_count(i)},
                        {"attends_next", !llama_hparams::is_masked_swa(4, LLAMA_SWA_TYPE_STANDARD, slots.pos_get(i), end)}});
    }
    std::cout << "CELLS " << name << " end=" << end << " " << rows.dump() << '\n';
}

static std::string row(llama_context * ctx, int pos) {
    auto * memory = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
    writer out;
    memory->get_swa()->state_write_range(out, 0, pos, pos + 1, 0);
    return out.data;
}

static std::string physical_state(llama_context * ctx) {
    auto * memory = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
    writer out;
    memory->get_swa()->state_write(out, 0, LLAMA_STATE_SEQ_FLAGS_KV_LAYOUT);
    require(out.data.size() >= 16, "missing physical state header");
    // Short contexts can allocate fewer cells; compare every retained index, position, sequence, head and K/V byte.
    memset(&out.data[8], 0, sizeof(uint32_t));
    return out.data;
}

// Another slot may reuse masked cells without preventing an ordinary checkpoint or slot restore.
static void unified_restore(llama_model * model) {
    for (int depth : {10, 256, 300, 512, 520, 600}) {
        for (llama_state_seq_flags flags : {LLAMA_STATE_SEQ_FLAGS_NONE, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY}) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 1024;
            cp.n_batch = cp.n_ubatch = 256;
            cp.n_seq_max = 2;
            cp.kv_unified = true;
            cp.swa_full = false;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
            cp.n_threads = cp.n_threads_batch = 2;
            hybrid::context_ptr ctx(llama_init_from_model(model, cp));
            require(bool(ctx), "unified context allocation failed");
            hybrid::prefill(ctx.get(), std::vector<llama_token>(depth, 3), 0);
            std::string checkpoint(llama_state_seq_get_size_ext(ctx.get(), 0, flags), '\0');
            require(llama_state_seq_get_data_ext(ctx.get(), (uint8_t *) checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(), "checkpoint save failed");
            llama_batch batch = llama_batch_init(3, 0, 1);
            for (int i = 0; i < 3; ++i) {
                batch.token[i] = 4;
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 1;
                batch.logits[i] = i == 2;
            }
            batch.n_tokens = 3;
            const int result = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(result == 0, "neighbor decode failed");
            auto * memory = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx.get()));
            const auto neighbor = snapshot(*memory, 1);
            require(llama_state_seq_set_data_ext(ctx.get(), (const uint8_t *) checkpoint.data(), checkpoint.size(), 0, flags) == checkpoint.size(),
                    "swa_unified_multislot_checkpoint_restore");
            require(snapshot(*memory, 1) == neighbor, "swa_unified_restore_preserves_neighbor");
            std::cout << "PASS swa_unified_multislot_checkpoint_restore depth=" << depth << " flags=" << flags << " bytes=" << checkpoint.size() << '\n';
        }
    }
}

// Real attention must match the same prefill partition after fresh and nonzero-prefix imports.
static void inference(const char * path) {
    tiny_model(path);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    hybrid::model_ptr model(llama_model_load_from_file(path, mp));
    require(bool(model), "cannot load tiny model");
    unified_restore(model.get());
    bool all_exact = true;
    for (int depth : {3, 4, 5, 19, 273, 510, 512, 514, 531}) {
        for (bool chain : {false, true}) {
            auto node = hybrid::make_context(model.get(), 1024);
            const int n_ctx = ((depth + 64 + 255) / 256) * 256;
            auto local = hybrid::make_context(model.get(), n_ctx);
            auto imported = hybrid::make_context(model.get(), n_ctx);
            std::vector<llama_token> tokens(depth);
            for (int i = 0; i < depth; ++i) { tokens[i] = 3 + (i * 7 + i / 3) % 5; }
            int start = 0;
            const int chunk = chain ? 7 : depth;
            while (start < depth) {
                const int end = std::min(depth, start + chunk);
                hybrid::prefill(node.get(), {tokens.begin() + start, tokens.begin() + end}, start);
                hybrid::prefill(local.get(), {tokens.begin() + start, tokens.begin() + end}, start);
                const auto delta = hybrid::export_range(node.get(), start, end);
                require(llama_state_seq_set_data_append(imported.get(), (const uint8_t *) delta.data(), delta.size(), 0, start, end) == delta.size(),
                        "tiny import failed");
                if (chain) { hybrid::load_state(node.get(), hybrid::save_state(node.get())); }
                require(physical_state(node.get()) == physical_state(local.get()) &&
                        physical_state(imported.get()) == physical_state(local.get()), "swa_physical_cells_and_head");
                start = end;
            }
            bool exact = true;
            double difference = 0;
            cells(node.get(), "node", depth);
            cells(local.get(), "local", depth);
            cells(imported.get(), "imported", depth);
            for (int pos = std::max(0, depth - 4); pos < depth; ++pos) {
                const bool source_equal = row(node.get(), pos) == row(local.get(), pos);
                const bool imported_equal = row(imported.get(), pos) == row(local.get(), pos);
                std::cout << "ROW position=" << pos << " source_equal=" << source_equal << " imported_equal=" << imported_equal << '\n';
                require(source_equal && imported_equal, "swa_retained_cell_bytes");
            }
            for (int i = 0; i < 64; ++i) {
                const llama_token token = 3 + i % 5;
                for (auto * ctx : {local.get(), imported.get()}) { hybrid::prefill(ctx, {token}, depth + i); }
                const auto * a = llama_get_logits_ith(local.get(), -1);
                const auto * b = llama_get_logits_ith(imported.get(), -1);
                exact &= memcmp(a, b, 8 * sizeof(float)) == 0;
                for (int j = 0; j < 8; ++j) { difference = std::max(difference, std::abs(double(a[j]) - b[j])); }
            }
            std::cout << "INFERENCE depth=" << depth << " chain=" << chain << " exact=" << exact << " max_diff=" << difference << '\n';
            all_exact &= exact;
        }
    }
    model.reset();
    require(std::remove(path) == 0, "cannot remove tiny model");
    require(all_exact, "swa_beyond_window_same_partition_logits");
}

int main(int argc, char ** argv) {
    try {
        llama_backend_init();
        for (auto arch : {LLM_ARCH_GEMMA4, LLM_ARCH_MUSE_GLIMMER}) {
            for (bool unified : {false, true}) { run(arch, unified); }
        }
        if (argc == 2) { inference(argv[1]); }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
