#include "hybrid-state.h"
#include "hybrid-scheduler.h"
#include "llama-ext.h"
#include "llama-model.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using hybrid::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void write(const std::string & path, const std::string & data) {
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), data.size());
    require(file.good(), "cannot write artifact");
}

static json header(const std::string & hash, size_t n0, size_t n1, const std::vector<llama_token> & tokens) {
    return {{"model_sha256", hash}, {"session_id", "golden"}, {"n0", n0}, {"n1", n1}, {"tokens", std::vector<llama_token>(tokens.begin() + n0, tokens.begin() + n1)}, {"prefix_hash_n1", hybrid::prefix_hash(tokens)}};
}

static void scheduler_tests() {
    // A later critical append cannot pass its session shadow; clients rotate even when one has only shadow work.
    hybrid::request_ticket a_shadow{"a", "one", false};
    hybrid::request_ticket a_later{"a", "one", true};
    hybrid::request_ticket b_critical{"b", "two", true};
    require(hybrid::next_request({&a_shadow, &a_later}, "") == &a_shadow, "session FIFO bypass");
    require(hybrid::next_request({&a_shadow, &a_later, &b_critical}, "a") == &b_critical, "critical priority lost");
    hybrid::request_ticket a{"a", "one", true};
    hybrid::request_ticket a_other{"a", "other", true};
    hybrid::request_ticket b{"b", "one", true};
    hybrid::request_ticket c{"c", "one", true};
    hybrid::request_ticket b_shadow{"b", "one", false};
    require(hybrid::next_request({&a, &a_other, &b_shadow}, "a") == &b_shadow, "shadow client starved by critical client");
    require(hybrid::next_request({&a_shadow, &a_other}, "") == &a_other, "within-client critical priority lost");
    require(hybrid::next_request({&a, &a_other, &b, &c}, "a") == &b, "client b starved");
    require(hybrid::next_request({&a, &a_other, &b, &c}, "b") == &c, "client c starved");
    require(hybrid::next_request({&a, &a_other, &b, &c}, "c") == &a, "round-robin wrap lost");
    require(hybrid::next_request({}, "c") == nullptr, "empty scheduler queue");
    std::cout << "PASS scheduler FIFO, critical priority and client round robin\n";
}

static void codec_tests(size_t max_ctx) {
    // Round trips accept the valid control; corrupt, truncated and incompatible envelopes are rejected before engine mutation.
    const json layout = {{"architecture", "qwen35"}, {"layer_count", 24}, {"embedding_size", 1024},
                         {"heads", 8}, {"kv_heads", 2}, {"recurrent_layout", json::array({true, false})},
                         {"vocab_sha256", std::string(64, 'a')}};
    const auto fingerprint = hybrid::layout_fingerprint(layout);
    json reordered = json::object();
    for (const auto & field : {"vocab_sha256", "recurrent_layout", "kv_heads", "heads", "embedding_size", "layer_count", "architecture"}) {
        reordered[field] = layout.at(field);
    }
    require(hybrid::layout_fingerprint(reordered) == fingerprint, "layout fingerprint depends on insertion order");
    for (const auto & field : {"architecture", "layer_count", "embedding_size", "heads", "kv_heads", "recurrent_layout", "vocab_sha256"}) {
        auto changed = layout;
        changed[field] = "changed";
        require(hybrid::layout_fingerprint(changed) != fingerprint, "different layout fingerprint matched");
    }
    require(hybrid::layout_fingerprint(layout) == fingerprint, "layout fingerprint unstable");
    std::cout << "PASS layout fingerprint structural mutation controls\n";
    const std::string hash(64, 'a');
    auto h = header(hash, 0, 1, {1});
    auto wire = hybrid::encode_delta(h, "abc");
    require(hybrid::read_delta(wire, hash, 0).state == "abc", "codec positive control");
    require(hybrid::prefix_hash({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty prefix hash");
    auto rejects = [&](const std::string & name, const std::string & input, const std::string & model_hash, int64_t n0, const std::string & expected) {
        try { hybrid::read_delta(input, model_hash, n0); }
        catch (const std::exception & e) {
            require(std::string(e.what()).find(expected) != std::string::npos, "wrong rejection reason");
            std::cout << "PASS " << name << ": " << e.what() << '\n';
            return;
        }
        throw std::runtime_error("accepted " + name);
    };
    rejects("wrong-model", wire, std::string(64, 'b'), 0, "model hash mismatch");
    rejects("overlap", wire, hash, 1, "position gap or overlap");
    auto gap_h = h; gap_h["n0"] = 2; gap_h["n1"] = 3;
    rejects("gap", hybrid::encode_delta(gap_h, "abc"), hash, 0, "position gap or overlap");
    rejects("truncated", wire.substr(0, wire.size() - 1), hash, 0, "truncated or trailing");
    rejects("trailing", wire + "x", hash, 0, "truncated or trailing");
    auto corrupt = wire; corrupt[corrupt.size() - 49] ^= 1;
    rejects("corrupt-section", corrupt, hash, 0, "checksum mismatch");
    corrupt = wire; corrupt[corrupt.size() - 1] ^= 1;
    rejects("corrupt-body-hash", corrupt, hash, 0, "checksum mismatch");
    uint32_t length = 0;
    for (int i = 0; i < 4; ++i) { length |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    auto bad_h = hybrid::parse_json(wire.substr(4, length));
    bad_h["kv_type"] = "f16";
    const std::string text = bad_h.dump();
    std::string bad;
    for (int i = 0; i < 4; ++i) { bad += (char) (text.size() >> (8 * i)); }
    bad += text + wire.substr(4 + length);
    rejects("layout", bad, hash, 0, "unsupported delta layout");
    auto wrong_tokens = h;
    wrong_tokens["tokens"] = json::array();
    rejects("token-count", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "token count mismatch");
    wrong_tokens["tokens"] = json::array({-1});
    rejects("negative-token", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "invalid delta token");
    auto wrong_prefix = h;
    wrong_prefix["prefix_hash_n1"] = std::string(64, '0');
    rejects("prefix-hash", hybrid::encode_delta(wrong_prefix, "abc"), hash, 0, "prefix hash mismatch");
    // Worst-case token IDs fit a bounded chunk of the configured context; exact-budget accepts and one-over rejects.
    const size_t count = std::min<size_t>(max_ctx, (hybrid::max_delta_header_bytes - 4096) / 11 + 1);
    std::vector<llama_token> maximum_tokens(count, INT32_MAX);
    auto maximum = header(hash, 0, maximum_tokens.size(), maximum_tokens);
    auto maximum_text = hybrid::encode_delta_header(maximum, hybrid::max_delta_body_bytes);
    require(maximum_text.size() <= std::min(hybrid::max_delta_header_bytes, 4096 + 11 * count), "maximum header budget");
    auto maximum_delta = hybrid::read_delta(hybrid::encode_delta(maximum, "abc"), hash, 0);
    require(maximum_delta.state == "abc" && maximum_delta.header.at("tokens").get<std::vector<llama_token>>() == maximum_tokens, "maximum header round trip");
    const size_t budget = std::min(hybrid::max_delta_header_bytes, 4096 + 11 * count);
    maximum["padding"] = "";
    const size_t unpadded = hybrid::encode_delta_header(maximum, 3).size();
    maximum["padding"] = std::string(budget - unpadded, 'x');
    auto exact_wire = hybrid::encode_delta(maximum, "abc");
    require(hybrid::read_delta(exact_wire, hash, 0).header.at("tokens").size() == count, "exact budget rejected");
    maximum["padding"].get_ref<std::string &>() += 'x';
    bool one_over_rejected = false;
    try { hybrid::encode_delta_header(maximum, 3); }
    catch (const std::exception & e) { one_over_rejected = std::string(e.what()) == "delta header exceeds range budget"; }
    require(one_over_rejected, "one-over writer accepted");
    uint32_t exact_size = 0;
    for (int i = 0; i < 4; ++i) { exact_size |= (uint32_t) (unsigned char) exact_wire[i] << (8 * i); }
    auto over_header = hybrid::parse_json(exact_wire.substr(4, exact_size));
    over_header["padding"].get_ref<std::string &>() += 'x';
    const auto over_text = over_header.dump();
    std::string over_wire;
    for (int i = 0; i < 4; ++i) { over_wire += (char) (over_text.size() >> (8 * i)); }
    over_wire += over_text + exact_wire.substr(4 + exact_size);
    rejects("one-over-reader", over_wire, hash, 0, budget == hybrid::max_delta_header_bytes ? "invalid delta header length" : "delta header exceeds range budget");
    auto padded = h;
    padded["padding"] = std::string(4096, 'x');
    bool rejected = false;
    try { hybrid::encode_delta_header(padded, hybrid::max_delta_body_bytes); }
    catch (const std::exception & e) { rejected = std::string(e.what()) == "delta header exceeds range budget"; }
    require(rejected, "writer header budget not enforced");
    padded = hybrid::parse_json(wire.substr(4, length));
    padded["padding"] = std::string(4096, 'x');
    const auto padded_text = padded.dump();
    std::string padded_wire;
    for (int i = 0; i < 4; ++i) { padded_wire += (char) (padded_text.size() >> (8 * i)); }
    padded_wire += padded_text + wire.substr(4 + length);
    rejects("reader-range-header-budget", padded_wire, hash, 0, "delta header exceeds range budget");
    std::string oversized_length;
    for (int i = 0; i < 4; ++i) { oversized_length += (char) ((1048577u >> (8 * i)) & 255); }
    rejects("reader-absolute-header-budget", oversized_length, hash, 0, "invalid delta header length");
    std::cout << "PASS test_header_budget_int32_max_ctx=" << max_ctx << " chunk=" << count << " exact_budget=" << budget << ": " << maximum_text.size() << " bytes with largest body metadata; writer/reader over-budget rejection\n";
    std::cout << "PASS codec positive and negative controls\n";
}

static std::vector<llama_token> prompt_tokens(llama_model * model, size_t n) {
    const std::string prompt = "User: Inspect the source files, preserve existing changes, and explain the test results.\nAssistant: I will inspect the implementation and check the invariant.\n";
    auto vocab = llama_model_get_vocab(model);
    std::vector<llama_token> base(256);
    int count = llama_tokenize(vocab, prompt.data(), prompt.size(), base.data(), base.size(), true, true);
    require(count > 0, "cannot tokenize test prompt");
    base.resize(count);
    std::vector<llama_token> out;
    for (size_t i = 0; i < n; ++i) { out.push_back(base[i % base.size()]); }
    return out;
}

static uint32_t read_u32(const std::string & data, size_t offset) {
    require(offset <= data.size() && data.size() - offset >= 4, "native fixture offset");
    uint32_t n = 0;
    for (int i = 0; i < 4; ++i) { n |= (uint32_t) (unsigned char) data[offset + i] << (8 * i); }
    return n;
}

static void put_u32(std::string & data, size_t offset, uint32_t n) {
    require(offset <= data.size() && data.size() - offset >= 4, "native fixture offset");
    for (int i = 0; i < 4; ++i) { data[offset + i] = (char) (n >> (8 * i)); }
}

static void append_negative_tests(llama_context * ctx, const std::string & raw, llama_pos n0, llama_pos n1) {
    // Invalid boundaries and native metadata must reject with byte-identical serialized prefix state, including after partial restore work.
    const auto before = hybrid::save_state(ctx);
    auto reject = [&](const char * name, const std::string & input, llama_pos start, llama_pos end) {
        require(llama_state_seq_set_data_append(ctx, reinterpret_cast<const uint8_t *>(input.data()), input.size(), 0, start, end) == 0, name);
        require(hybrid::save_state(ctx) == before, "rejected append changed prefix state");
        std::cout << "PASS " << name << " with unchanged prefix state\n";
    };
    reject("raw-gap", raw, n0 + 1, n1);
    reject("raw-overlap", raw, n0 - 1, n1);
    reject("raw-end-mismatch", raw, n0, n1 + 1);
    reject("raw-truncated", raw.substr(0, raw.size() - 1), n0, n1);
    auto corrupt = raw;
    corrupt[0] ^= 1;
    reject("raw-magic", corrupt, n0, n1);
    corrupt = raw;
    put_u32(corrupt, 8, 0);
    reject("raw-stream-layout", corrupt, n0, n1);
    corrupt = raw;
    put_u32(corrupt, 16, n0 + 1);
    reject("raw-position-gap", corrupt, n0, n1);

    // These fixture controls locate the pinned native attention header for ordinary or M-RoPE cell metadata.
    const uint32_t cells = read_u32(raw, 12);
    require(cells == (uint32_t) (n1 - n0), "range cell count mismatch");
    size_t stride = 0;
    for (size_t candidate : {12, 24}) {
        bool matches = true;
        for (uint32_t i = 0; i < cells; ++i) {
            matches = matches && read_u32(raw, 16 + i * candidate) == (uint32_t) (n0 + i);
            matches = matches && read_u32(raw, 20 + i * candidate) == 1;
        }
        if (matches) { stride = candidate; }
    }
    require(stride != 0, "unsupported native fixture cell metadata");
    const size_t layer_offset = 16 + cells * stride + 4;
    corrupt = raw;
    put_u32(corrupt, layer_offset, read_u32(raw, layer_offset) + 1);
    reject("raw-layer-layout", corrupt, n0, n1);
}

static void run_model(const std::string & model_path, const std::string & directory, size_t depth, size_t max_ctx) {
    require(depth >= 256 && depth + 64 <= max_ctx, "depth exceeds max-ctx continuation capacity");
    const auto hash = hybrid::file_hash(model_path);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    hybrid::model_ptr model(llama_model_load_from_file(model_path.c_str(), mp));
    require(bool(model), "model load failed");
    // The actual model-to-layout projection detects a changed embedding geometry, not just changed JSON input.
    const auto layout = hybrid::model_layout(model.get());
    const auto embedding_size = model->hparams.n_embd;
    model->hparams.n_embd = embedding_size + 1;
    const auto changed_layout = hybrid::model_layout(model.get());
    model->hparams.n_embd = embedding_size;
    require(layout.at("embedding_size") == embedding_size && changed_layout.at("embedding_size") == embedding_size + 1,
            "model layout projection ignored embedding geometry");
    require(hybrid::layout_fingerprint(layout) != hybrid::layout_fingerprint(changed_layout), "changed model geometry matched fingerprint");
    std::cout << "PASS model layout projection: changed embedding geometry differs\n";
    auto source = hybrid::make_context(model.get(), depth + 256);
    auto restored = hybrid::make_context(model.get(), depth + 256);
    const auto tokens = prompt_tokens(model.get(), depth);
    const size_t n0 = depth - 129;
    const size_t n1 = depth - 1;
    hybrid::prefill(source.get(), {tokens.begin(), tokens.begin() + n0}, 0);
    const auto prefix = hybrid::save_state(source.get());
    require(llama_state_seq_get_data_range(source.get(), nullptr, 0, 0, 0, n0 - 1, 0) == 0, "accepted stale recurrent endpoint");
    require(llama_state_seq_get_data_range(source.get(), nullptr, 0, 0, n0, n0, 0) == 0, "accepted empty range");
    require(llama_state_seq_get_data_range(source.get(), nullptr, 0, 0, -1, n0, 0) == 0, "accepted negative boundary");
    require(llama_state_seq_get_data_range(source.get(), nullptr, 0, 0, 0, n0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0, "accepted unsupported flags");
    require(hybrid::save_state(source.get()) == prefix, "range rejection changed source state");
    hybrid::prefill(source.get(), {tokens.begin() + n0, tokens.begin() + n1}, n0);
    auto raw = hybrid::export_range(source.get(), n0, n1);
    auto wire = hybrid::encode_delta(header(hash, n0, n1, {tokens.begin(), tokens.begin() + n1}), raw);
    auto parsed = hybrid::read_delta(wire, hash, n0);
    require(parsed.state == raw, "delta changed range bytes");
    require(raw.size() < hybrid::save_state(source.get()).size(), "range export did not reduce attention data");
    auto full_raw = hybrid::export_range(source.get(), 0, n1);
    hybrid::load_state(restored.get(), prefix);
    append_negative_tests(restored.get(), parsed.state, n0, n1);
    require(llama_state_seq_set_data_append(restored.get(), reinterpret_cast<const uint8_t *>(parsed.state.data()), parsed.state.size(), 0, n0, n1) == parsed.state.size(), "incremental append failed");
    require(hybrid::save_state(restored.get()) == full_raw, "appended state differs from full source state");
    // A prefix plus range append and held final token matches 64 tokens from an independent plain local prefill.
    hybrid::prefill(restored.get(), {tokens.back()}, n1);
    hybrid::load_state(source.get(), {});
    hybrid::prefill(source.get(), tokens, 0);
    std::vector<llama_token> generated;
    for (size_t i = 0; i < 64; ++i) {
        auto local = hybrid::greedy(source.get(), model.get());
        auto imported = hybrid::greedy(restored.get(), model.get());
        require(local == imported, "greedy mismatch");
        generated.push_back(local);
        hybrid::prefill(source.get(), {local}, depth + i);
        hybrid::prefill(restored.get(), {imported}, depth + i);
    }
    if (!directory.empty()) {
        std::filesystem::create_directories(directory);
        const std::string stem = directory + "/prompt-" + std::to_string(depth);
        write(stem + ".prefix.state", prefix);
        write(stem + ".delta", wire);
        auto full_wire = hybrid::encode_delta(header(hash, 0, n1, {tokens.begin(), tokens.begin() + n1}), full_raw);
        write(stem + ".full.delta", full_wire);
        write(stem + ".tokens.json", json({{"model_sha256", hash}, {"n0", n0}, {"n1", n1}, {"tokens", tokens}, {"held_token", tokens.back()}, {"greedy_64", generated}, {"prefix_hash_n0", hybrid::prefix_hash({tokens.begin(), tokens.begin() + n0})}, {"prefix_hash_n1", hybrid::prefix_hash({tokens.begin(), tokens.begin() + n1})}}).dump(2) + "\n");
    }
    std::cout << "PASS depth=" << depth << " n0=" << n0 << " n1=" << n1 << " range_bytes=" << raw.size() << " full_bytes=" << full_raw.size() << " greedy=64/64 (incremental range append vs independent local prefill)\n";
}

int main(int argc, char ** argv) {
    try {
        size_t max_ctx = 16384;
        std::vector<std::string> positional;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--max-ctx" && i + 1 < argc) { max_ctx = std::stoul(argv[++i]); }
            else { positional.push_back(argv[i]); }
        }
        require(max_ctx >= 256 && max_ctx <= INT32_MAX - 256, "invalid max-ctx");
        scheduler_tests();
        codec_tests(max_ctx);
        if (positional.size() == 1 && positional[0] == "--codec") { return 0; }
        if (positional.size() < 2 || positional.size() > 3) {
            std::cerr << "usage: llama-hybrid-check --codec | MODEL.gguf GOLDEN_DIR [DEPTH] [--max-ctx N]\n";
            return 2;
        }
        llama_backend_init();
        const auto directory = positional[1] == "-" ? "" : positional[1];
        if (positional.size() == 3) {
            run_model(positional[0], directory, std::stoul(positional[2]), max_ctx);
        } else {
            for (size_t depth : {512, 2048, 8192}) { run_model(positional[0], directory, depth, max_ctx); }
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
