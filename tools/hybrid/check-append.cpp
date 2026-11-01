#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-model.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open golden file");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static std::string replace_header(const std::string & wire, const hybrid::json & header) {
    uint32_t old_size = 0;
    for (int i = 0; i < 4; ++i) { old_size |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    auto text = header.dump();
    std::string result;
    for (int i = 0; i < 4; ++i) { result += (char) (text.size() >> (8 * i)); }
    return result + text + wire.substr(4 + old_size);
}

static void run(llama_model * model, const std::string & model_hash, const std::string & stem) {
    auto metadata = hybrid::parse_json(read_file(stem + ".tokens.json"));
    require(metadata.at("model_sha256") == model_hash, "golden model mismatch");
    auto tokens = metadata.at("tokens").get<std::vector<llama_token>>();
    const auto n0 = metadata.at("n0").get<llama_pos>();
    const auto n1 = metadata.at("n1").get<llama_pos>();
    require(n0 > 0 && n1 == (llama_pos) tokens.size() - 1 && n0 < n1, "invalid golden boundaries");
    const auto wire = read_file(stem + ".delta");
    const auto delta = hybrid::read_delta(wire, model_hash, n0);
    {
        // A fresh slot also preserves its empty state on a late failure, then accepts a valid initial import.
        auto fresh = hybrid::make_context(model, tokens.size() + 256);
        const auto empty = hybrid::save_state(fresh.get());
        const auto full = hybrid::read_delta(read_file(stem + ".full.delta"), model_hash, 0);
        require(llama_state_seq_set_data_append(fresh.get(), reinterpret_cast<const uint8_t *>(full.state.data()),
                full.state.size() - 1, 0, 0, n1) == 0, "fresh truncated state accepted");
        require(hybrid::save_state(fresh.get()) == empty, "fresh failed import changed slot");
        require(llama_state_seq_set_data_append(fresh.get(), reinterpret_cast<const uint8_t *>(full.state.data()),
                full.state.size(), 0, 0, n1) == full.state.size(), "fresh valid import rejected");
        require(llama_memory_seq_pos_max(llama_get_memory(fresh.get()), 0) + 1 == n1, "fresh imported endpoint mismatch");
        std::cout << "PASS fresh slot rollback and import\n";
    }
    auto imported = hybrid::make_context(model, tokens.size() + 256);
    auto local = hybrid::make_context(model, tokens.size() + 256);
    hybrid::load_state(imported.get(), read_file(stem + ".prefix.state"));
    auto before = hybrid::save_state(imported.get());
    require(llama_memory_seq_pos_max(llama_get_memory(imported.get()), 0) + 1 == n0, "prefix endpoint mismatch");
    std::vector<llama_token> merged(tokens.begin(), tokens.begin() + n0);
    auto suffix = delta.header.at("tokens").get<std::vector<llama_token>>();
    merged.insert(merged.end(), suffix.begin(), suffix.end());
    require(hybrid::prefix_hash(merged) == delta.header.at("prefix_hash_n1"), "merged prefix mismatch");

    // Envelope rejection happens before import, and the exact serialized destination remains unchanged.
    auto reject_wire = [&](const char * name, const std::string & candidate, const std::string & expected) {
        bool rejected = false;
        try {
            auto parsed = hybrid::read_delta(candidate, model_hash, n0);
            auto combined = std::vector<llama_token>(tokens.begin(), tokens.begin() + n0);
            auto addition = parsed.header.at("tokens").get<std::vector<llama_token>>();
            combined.insert(combined.end(), addition.begin(), addition.end());
            if (hybrid::prefix_hash(combined) != parsed.header.at("prefix_hash_n1")) {
                throw std::runtime_error("prefix hash mismatch");
            }
        } catch (const std::exception & error) {
            require(std::string(error.what()).find(expected) != std::string::npos, "wrong envelope rejection reason");
            rejected = true;
        }
        require(rejected, "invalid envelope was accepted");
        require(hybrid::save_state(imported.get()) == before, "envelope rejection changed destination");
        std::cout << "PASS " << name << " slot unchanged\n";
    };
    auto h = delta.header;
    h["model_sha256"] = std::string(64, 'f');
    reject_wire("wrong-model", replace_header(wire, h), "model hash mismatch");
    h = delta.header; h["n0"] = n0 + 1;
    reject_wire("gap", replace_header(wire, h), "position gap or overlap");
    h["n0"] = n0 - 1;
    reject_wire("overlap", replace_header(wire, h), "position gap or overlap");
    reject_wire("truncated-envelope", wire.substr(0, wire.size() - 1), "truncated or trailing");
    auto bad = wire; bad[bad.size() - 48] ^= 1;
    reject_wire("section-checksum", bad, "checksum mismatch");
    bad = wire; bad.back() ^= 1;
    reject_wire("body-checksum", bad, "checksum mismatch");
    h = delta.header; h["prefix_hash_n1"] = std::string(64, 'f');
    reject_wire("prefix-hash", replace_header(wire, h), "prefix hash mismatch");
    h = delta.header; h["kv_type"] = "f16";
    reject_wire("declared-layout", replace_header(wire, h), "unsupported delta layout");

    // These probes reach the native reader, including a late recurrent failure after KV metadata changed.
    auto reject_raw = [&](const char * name, const std::string & state, llama_pos start, llama_pos end) {
        const size_t result = llama_state_seq_set_data_append(imported.get(),
            reinterpret_cast<const uint8_t *>(state.data()), state.size(), 0, start, end);
        require(result == 0, "native invalid state was accepted");
        require(hybrid::save_state(imported.get()) == before, "native rejection changed destination");
        std::cout << "PASS " << name << " slot unchanged\n";
    };
    reject_raw("raw-gap", delta.state, n0 + 1, n1);
    reject_raw("raw-overlap", delta.state, n0 - 1, n1);
    reject_raw("raw-truncated", delta.state.substr(0, delta.state.size() - 1), n0, n1);
    reject_raw("raw-trailing", delta.state + "x", n0, n1);
    reject_raw("raw-endpoint", delta.state, n0, n1 + 1);
    auto raw = delta.state;
    auto set32 = [&](size_t at, uint32_t value) {
        require(at + 4 <= raw.size(), "fixture offset out of range");
        memcpy(&raw[at], &value, 4);
    };
    auto get32 = [&](size_t at) {
        uint32_t value;
        require(at + 4 <= raw.size(), "fixture offset out of range");
        memcpy(&value, raw.data() + at, 4);
        return value;
    };
    require(get32(8) == 1, "fixture must have one native stream");
    const auto count = get32(12);
    require(count == (uint32_t) (n1 - n0), "fixture cell count mismatch");
    set32(16, n0 + 1);
    reject_raw("raw-cell-gap", raw, n0, n1);
    raw = delta.state;
    const bool ext = model->hparams.n_pos_per_embd() > 1 || model->hparams.ple_n_heads > 0;
    size_t offset = 16 + count * (12 + (ext ? 12 : 0));
    require(get32(offset) == 0, "fixture must have untransposed values");
    const auto layers = get32(offset + 4);
    set32(offset + 4, layers + 1);
    reject_raw("raw-layer-layout", raw, n0, n1);
    raw = delta.state;
    offset += 16;
    for (size_t i = 0; i < 2 * layers; ++i) {
        require(offset + 12 <= raw.size(), "fixture tensor header out of range");
        uint64_t row;
        memcpy(&row, raw.data() + offset + 4, 8);
        require(row <= (raw.size() - offset - 12) / count, "fixture tensor out of range");
        offset += 12 + row * count;
    }
    require(get32(offset) == 1, "fixture must have one recurrent state");
    set32(offset + 4, n1);
    reject_raw("raw-recurrent-endpoint", raw, n0, n1);

    // Incremental prefix + delta + held token must match a separate plain local prefill for 64 greedy tokens.
    require(llama_state_seq_set_data_append(imported.get(), reinterpret_cast<const uint8_t *>(delta.state.data()),
            delta.state.size(), 0, n0, n1) == delta.state.size(), "valid append rejected");
    hybrid::prefill(imported.get(), {tokens.back()}, n1);
    hybrid::prefill(local.get(), tokens, 0);
    auto expected = metadata.at("greedy_64").get<std::vector<llama_token>>();
    require(expected.size() == 64, "golden must have 64 expected tokens");
    for (size_t i = 0; i < 64; ++i) {
        auto a = hybrid::greedy(imported.get(), model);
        auto b = hybrid::greedy(local.get(), model);
        require(a == b, "incremental append/local greedy mismatch");
        require(a == expected[i], "incremental append/golden greedy mismatch");
        hybrid::prefill(imported.get(), {a}, tokens.size() + i);
        hybrid::prefill(local.get(), {b}, tokens.size() + i);
    }
    std::cout << "PASS " << stem << " incremental append + held token greedy=64/64\n";
}

static void chunk_tests(llama_model * model, const std::string & hash) {
    auto source = hybrid::make_context(model, 512);
    auto single = hybrid::make_context(model, 512);
    auto dest = hybrid::make_context(model, 512);
    const std::vector<llama_token> prefix {100, 101, 102, 103};
    hybrid::prefill(source.get(), prefix, 0);
    hybrid::prefill(single.get(), prefix, 0);
    hybrid::prefill(dest.get(), prefix, 0);
    const auto before = hybrid::save_state(dest.get());
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto * logits_ptr = llama_get_logits_ith(dest.get(), -1);
    const std::vector<float> logits(logits_ptr, logits_ptr + vocab);
    auto unchanged = [&] {
        require(hybrid::save_state(dest.get()) == before, "chunk failure changed cells or state");
        require(memcmp(logits.data(), llama_get_logits_ith(dest.get(), -1), vocab * sizeof(float)) == 0, "chunk failure changed next-token logits");
    };
    std::vector<llama_token> all = prefix;
    std::vector<std::string> wires, bodies, full_bodies;
    std::vector<hybrid::json> headers;
    for (int i = 0; i < 4; ++i) {
        const int n0 = all.size();
        const std::vector<llama_token> suffix {110 + i * 4, 111 + i * 4, 112 + i * 4, 113 + i * 4};
        hybrid::prefill(source.get(), suffix, n0);
        all.insert(all.end(), suffix.begin(), suffix.end());
        hybrid::json h = {{"model_sha256", hash}, {"session_id", "chunks"}, {"n0", n0}, {"n1", all.size()},
            {"tokens", suffix}, {"prefix_hash_n1", hybrid::prefix_hash(all)}, {"transfer_id", "test-transfer"},
            {"transfer_n0", prefix.size()}, {"transfer_n1", prefix.size() + 16}, {"chunk_index", i}, {"chunk_count", 4}, {"fixed_state", i == 3}};
        bodies.push_back(hybrid::export_range(source.get(), n0, all.size(), i == 3));
        full_bodies.push_back(hybrid::export_range(source.get(), n0, all.size()));
        wires.push_back(hybrid::encode_delta(h, bodies.back()));
        headers.push_back(h);
    }
    auto stage = [&](hybrid::delta_import & pending, const std::string & wire) {
        return pending.stage(dest.get(), 0, wire, hash, "chunks", prefix, 512, vocab);
    };
    // Dropping any proper chunk prefix leaves serialized cells/state and every next-token logit unchanged.
    for (size_t k : {1, 2, 3}) {
        hybrid::delta_import pending;
        for (size_t i = 0; i < k; ++i) { require(!stage(pending, wires[i]), "early chunk committed"); unchanged(); }
        pending.clear();
        unchanged();
        std::cout << "PASS chunk_interruption_k=" << k << " cells/state/logits unchanged\n";
    }
    auto reject = [&](const char * name, const std::vector<std::string> & candidates, const char * reason, bool apply = false) {
        hybrid::delta_import pending;
        bool rejected = false;
        try {
            for (const auto & wire : candidates) { stage(pending, wire); }
            if (apply) { pending.apply(dest.get(), 0); }
        } catch (const std::exception & e) {
            require(std::string(e.what()).find(reason) != std::string::npos, "wrong chunk rejection reason");
            rejected = true;
        }
        require(rejected, name);
        try { unchanged(); }
        catch (const std::exception & error) { throw std::runtime_error(std::string(name) + ": " + error.what()); }
        std::cout << "PASS " << name << " cells/state/logits unchanged\n";
    };
    // Malformed framing, identity, order and fixed-state placement reject before the destination can change.
    reject("chunk_out_of_order", {wires[1]}, "position gap or overlap");
    reject("chunk_duplicate", {wires[0], wires[0]}, "position gap or overlap");
    reject("chunk_missing_middle", {wires[0], wires[2]}, "position gap or overlap");
    reject("chunk_missing_final", {wires[0], wires[1], wires[2]}, "missing final chunk", true);
    auto h = headers[0];
    auto parsed = hybrid::read_delta(wires[0], hash, prefix.size()).header;
    parsed["fixed_state"] = true;
    reject("chunk_early_state_flag", {replace_header(wires[0], parsed)}, "fixed state must occur only");
    reject("chunk_early_native_state", {hybrid::encode_delta(headers[0], full_bodies[0])}, "invalid native chunk");
    parsed = hybrid::read_delta(wires[3], hash, 16).header;
    parsed["fixed_state"] = false;
    reject("chunk_final_state_flag_missing", {wires[0], wires[1], wires[2], replace_header(wires[3], parsed)}, "fixed state must occur only");
    const auto no_state = hybrid::export_range(source.get(), 16, 20, false);
    reject("chunk_final_native_state_missing", {wires[0], wires[1], wires[2], hybrid::encode_delta(headers[3], no_state)}, "invalid native chunk");
    parsed = hybrid::read_delta(wires[1], hash, 8).header; parsed["version"] = 1;
    reject("chunk_mixed_version", {wires[0], replace_header(wires[1], parsed)}, "unsupported delta version");
    h = headers[1]; h["transfer_id"] = "other";
    reject("chunk_mixed_transfer", {wires[0], hybrid::encode_delta(h, bodies[1])}, "chunk transfer mismatch");
    auto corrupt = wires[1]; corrupt.back() ^= 1;
    reject("chunk_checksum", {wires[0], corrupt}, "checksum mismatch");
    // A final recurrent endpoint mismatch reaches apply after all KV metadata changes; rollback also preserves logits.
    auto late = bodies[3];
    const size_t fixed_offset = no_state.size();
    const int32_t wrong_pos = 20;
    require(fixed_offset + 8 < late.size(), "no recurrent fixture");
    memcpy(&late[fixed_offset + 4], &wrong_pos, sizeof(wrong_pos));
    reject("chunk_late_endpoint_rollback", {wires[0], wires[1], wires[2], hybrid::encode_delta(headers[3], late)}, "Native delta chunks rejected", true);
    // Identical source evaluation yields byte-identical chunked and single-shot imports, then identical held-token logits.
    const auto full = hybrid::export_range(source.get(), prefix.size(), all.size());
    require(llama_state_seq_set_data_append(single.get(), (const uint8_t *) full.data(), full.size(), 0, prefix.size(), all.size()) == full.size(), "single import failed");
    hybrid::delta_import pending;
    for (size_t i = 0; i < wires.size(); ++i) { require(stage(pending, wires[i]) == (i == 3), "wrong final marker"); unchanged(); }
    pending.apply(dest.get(), 0);
    require(hybrid::save_state(dest.get()) == hybrid::save_state(single.get()), "chunk/single bytes differ");
    hybrid::prefill(single.get(), {130}, all.size());
    hybrid::prefill(dest.get(), {130}, all.size());
    require(memcmp(llama_get_logits_ith(single.get(), -1), llama_get_logits_ith(dest.get(), -1), vocab * sizeof(float)) == 0, "chunk/single held-token logits differ");
    std::cout << "PASS chunk_single_shot_byte_equivalence and held-token logits\n";
}

int main(int argc, char ** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: llama-hybrid-check-append MODEL.gguf GOLDEN_STEM [GOLDEN_STEM ...]\n";
            return 2;
        }
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], params));
        require(bool(model), "cannot load model");
        const auto hash = hybrid::file_hash(argv[1]);
        if (argc == 3 && std::string(argv[2]) == "--chunks") { chunk_tests(model.get(), hash); return 0; }
        for (int i = 2; i < argc; ++i) { run(model.get(), hash, argv[i]); }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
