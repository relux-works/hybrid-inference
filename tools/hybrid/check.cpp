#include "hybrid-state.h"
#include "hybrid-scheduler.h"
#include "llama-ext.h"
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
    // A later critical append cannot pass the same session's shadow; other sessions prefer critical work and rotate clients.
    hybrid::request_ticket a_shadow{"a", "one", false};
    hybrid::request_ticket a_later{"a", "one", true};
    hybrid::request_ticket b_critical{"b", "two", true};
    require(hybrid::next_request({&a_shadow, &a_later}, "") == &a_shadow, "session FIFO bypass");
    require(hybrid::next_request({&a_shadow, &a_later, &b_critical}, "") == &b_critical, "critical priority lost");
    hybrid::request_ticket a{"a", "one", true};
    hybrid::request_ticket a_other{"a", "other", true};
    hybrid::request_ticket b{"b", "one", true};
    hybrid::request_ticket c{"c", "one", true};
    require(hybrid::next_request({&a, &a_other, &b, &c}, "a") == &b, "client b starved");
    require(hybrid::next_request({&a, &a_other, &b, &c}, "b") == &c, "client c starved");
    require(hybrid::next_request({&a, &a_other, &b, &c}, "c") == &a, "round-robin wrap lost");
    require(hybrid::next_request({}, "c") == nullptr, "empty scheduler queue");
    std::cout << "PASS scheduler FIFO, critical priority and client round robin\n";
}

static void codec_tests() {
    // Round trips accept the valid control; corrupt, truncated and incompatible envelopes are rejected before engine mutation.
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

static void run_model(const std::string & model_path, const std::string & directory, size_t depth) {
    require(depth >= 256 && depth <= 8192, "depth must be 256-8192");
    const auto hash = hybrid::file_hash(model_path);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    hybrid::model_ptr model(llama_model_load_from_file(model_path.c_str(), mp));
    require(bool(model), "model load failed");
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
    // A fresh context restores the full range, evaluates the held final prompt token, then matches 64 local-prefill greedy tokens.
    auto full_raw = hybrid::export_range(source.get(), 0, n1);
    hybrid::load_state(restored.get(), full_raw);
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
    std::cout << "PASS depth=" << depth << " n0=" << n0 << " n1=" << n1 << " range_bytes=" << raw.size() << " full_bytes=" << full_raw.size() << " greedy=64/64 (full-range restore; incremental append awaits B)\n";
}

int main(int argc, char ** argv) {
    try {
        scheduler_tests();
        codec_tests();
        if (argc == 2 && std::string(argv[1]) == "--codec") { return 0; }
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: llama-hybrid-check --codec | MODEL.gguf GOLDEN_DIR [DEPTH]\n";
            return 2;
        }
        llama_backend_init();
        if (argc == 4) {
            run_model(argv[1], argv[2], std::stoul(argv[3]));
        } else {
            for (size_t depth : {512, 2048, 8192}) { run_model(argv[1], argv[2], depth); }
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
