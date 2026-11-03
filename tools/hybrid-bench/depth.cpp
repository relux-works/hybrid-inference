#include "hybrid-state.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using hybrid::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static json read_json(const std::string & path) {
    std::ifstream in(path);
    require(in.good(), "cannot read JSON");
    return hybrid::parse_json({std::istreambuf_iterator<char>(in), {}}, 64 * 1024 * 1024);
}

static bool same_json(const json & a, const json & b) {
    return nlohmann::json(a) == nlohmann::json(b);
}

static void write_json(const std::string & path, const json & value) {
    require(!std::filesystem::exists(path), "output already exists");
    std::ofstream out(path);
    out << value.dump(2) << '\n';
    out.close();
    require(out.good(), "cannot write JSON");
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    require(n < 0, "empty tokenization");
    std::vector<llama_token> tokens(-n);
    require(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true) == -n, "tokenization failed");
    return tokens;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, size_t begin, size_t end, int batch_size, bool all_logits = false) {
    for (size_t offset = begin; offset < end; offset += batch_size) {
        const size_t n = std::min<size_t>(batch_size, end - offset);
        auto batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (size_t i = 0; i < n; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = offset + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = all_logits || i == n - 1;
        }
        const int code = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(code == 0, "decode failed");
    }
}

static json binding(const json & fixture) {
    const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
    const auto tail = fixture.at("tail").get<std::vector<llama_token>>();
    return {{"model_sha256", fixture.at("model_sha256")}, {"tokens_sha256", hybrid::prefix_hash(tokens)},
            {"tail_sha256", hybrid::prefix_hash(tail)}, {"window", fixture.value("window", std::string("after"))}, {"needle_mode", fixture.at("needle_mode")}, {"probe_tokens", fixture.at("probe_tokens")}, {"source", fixture.at("source")}, {"settings", fixture.at("settings")}, {"needles", fixture.at("needles")}};
}

static void save_state(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens, const json & identity) {
    require(!std::filesystem::exists(path) && !std::filesystem::exists(path + ".json"), "state already exists");
    const size_t n = llama_state_seq_save_file(ctx, path.c_str(), 0, tokens.data(), tokens.size());
    require(n > 0, "state save failed");
    write_json(path + ".json", {{"binding", identity}, {"state_sha256", hybrid::file_hash(path)}, {"tokens_sha256", hybrid::prefix_hash(tokens)}, {"bytes", std::filesystem::file_size(path)}});
}

static void verify_state(const std::string & path, const json & identity, const std::vector<llama_token> & tokens, bool checksum) {
    const auto record = read_json(path + ".json");
    require(same_json(record.at("binding"), identity), "state document or YaRN mismatch");
    require(record.at("bytes") == std::filesystem::file_size(path), "state size mismatch");
    require(!checksum || record.at("state_sha256") == hybrid::file_hash(path), "state checksum mismatch");
    require(record.at("tokens_sha256") == hybrid::prefix_hash(tokens), "state document mismatch");
}

static void export_transfer(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens, int batch_size, size_t chunk_size, const json & identity) {
    require(!std::filesystem::exists(path) && !std::filesystem::exists(path + ".json"), "state already exists");
    std::ofstream out(path, std::ios::binary);
    require(chunk_size >= 256 && chunk_size <= 8192 && chunk_size % 256 == 0, "invalid transfer chunk size");
    const size_t count = (tokens.size() + chunk_size - 1) / chunk_size;
    for (size_t begin = 0, index = 0; begin < tokens.size(); begin += chunk_size, ++index) {
        const size_t end = std::min(tokens.size(), begin + chunk_size);
        decode(ctx, tokens, begin, end, batch_size);
        const std::vector<llama_token> suffix(tokens.begin() + begin, tokens.begin() + end), prefix(tokens.begin(), tokens.begin() + end);
        json header = {{"model_sha256", identity.at("model_sha256")}, {"session_id", "depth-quality"},
            {"n0", begin}, {"n1", end}, {"tokens", suffix}, {"prefix_hash_n1", hybrid::prefix_hash(prefix)},
            {"transfer_id", "depth-quality"}, {"transfer_n0", 0}, {"transfer_n1", tokens.size()},
            {"chunk_index", index}, {"chunk_count", count}, {"fixed_state", end == tokens.size()}};
        const auto wire = hybrid::encode_delta(header, hybrid::export_range(ctx, begin, end, end == tokens.size()));
        const uint64_t size = wire.size();
        for (int i = 0; i < 8; ++i) { out.put(char(size >> (8 * i))); }
        out.write(wire.data(), wire.size());
        require(out.good(), "transfer write failed");
    }
    out.close();
    require(out.good(), "transfer close failed");
    write_json(path + ".json", {{"binding", identity}, {"state_sha256", hybrid::file_hash(path)},
        {"tokens_sha256", hybrid::prefix_hash(tokens)}, {"bytes", std::filesystem::file_size(path)}, {"format", "hybrid-delta-chain-v2"}, {"chunk_count", count}});
}

static json import_transfer(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens, const json & identity, int vocab, bool save_snapshot) {
    verify_state(path, identity, tokens, true);
    const auto transfer = read_json(path + ".json");
    require(transfer.at("format") == "hybrid-delta-chain-v2", "wrong transfer format");
    std::ifstream in(path, std::ios::binary);
    hybrid::delta_import pending;
    bool final = false;
    while (in.peek() != std::char_traits<char>::eof()) {
        require(!final, "trailing transfer chunk");
        uint64_t size = 0;
        for (int i = 0; i < 8; ++i) {
            const int byte = in.get();
            require(byte != std::char_traits<char>::eof(), "truncated transfer size");
            size |= uint64_t(uint8_t(byte)) << (8 * i);
        }
        require(size > 0 && size <= hybrid::max_delta_body_bytes + hybrid::max_delta_header_bytes + 52, "invalid transfer chunk size");
        std::string wire(size, '\0');
        require(bool(in.read(&wire[0], size)), "truncated transfer chunk");
        final = pending.stage(ctx, 0, std::move(wire), identity.at("model_sha256"), "depth-quality", {}, llama_n_ctx(ctx), vocab);
    }
    require(final && pending.imported_tokens() == tokens, "incomplete transfer document");
    pending.apply(ctx, 0);
    pending.clear();
    // Remove the transfer before saving the reusable post-context snapshot: one state file on disk.
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".json");
    if (save_snapshot) { save_state(ctx, path, tokens, identity); }
    return transfer;
}

static void load_state(llama_context * ctx, const std::string & path, const std::vector<llama_token> & tokens, const json & identity, bool checksum) {
    verify_state(path, identity, tokens, checksum);
    std::vector<llama_token> loaded(tokens.size());
    size_t count = 0;
    llama_memory_clear(llama_get_memory(ctx), true);
    require(llama_state_seq_load_file(ctx, path.c_str(), 0, loaded.data(), loaded.size(), &count) > 0 && count == tokens.size() && loaded == tokens, "state import failed");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == int64_t(tokens.size() - 1), "restored state depth mismatch");
}

static void write_logits(llama_context * ctx, const std::vector<llama_token> & tokens, size_t begin, int batch_size, int vocab, const std::string & path) {
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4, "float32 little-endian host required");
    require(!std::filesystem::exists(path), "logits already exist");
    std::ofstream out(path, std::ios::binary);
    for (size_t i = begin; i < tokens.size(); i += batch_size) {
        const size_t end = std::min(tokens.size(), i + batch_size);
        decode(ctx, tokens, i, end, batch_size, true);
        for (size_t j = 0; j < end - i; ++j) {
            const float * row = llama_get_logits_ith(ctx, j);
            require(row, "missing logits");
            for (int k = 0; k < vocab; ++k) { require(std::isfinite(row[k]), "nonfinite logits"); }
            out.write(reinterpret_cast<const char *>(row), vocab * sizeof(float));
        }
    }
    out.close();
    require(out.good(), "logit write failed");
}

static json metrics(const std::vector<float> & a, const std::vector<float> & b) {
    require(!a.empty() && a.size() == b.size(), "logit shape mismatch");
    double za = 0, zb = 0, maximum = 0;
    const double ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite logit");
        za += std::exp(a[i] - ma);
        zb += std::exp(b[i] - mb);
        maximum = std::max(maximum, std::abs(double(a[i]) - b[i]));
    }
    za = ma + std::log(za);
    zb = mb + std::log(zb);
    double ab = 0, ba = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double p = a[i] - za, q = b[i] - zb;
        ab += std::exp(p) * (p - q);
        ba += std::exp(q) * (q - p);
    }
    return {{"a_to_b_kl_nats", std::max(0.0, ab)}, {"b_to_a_kl_nats", std::max(0.0, ba)},
            {"top1_agreed", std::max_element(a.begin(), a.end()) - a.begin() == std::max_element(b.begin(), b.end()) - b.begin()}, {"max_abs_logit_difference", maximum}};
}

static void compare(const json & job) {
    const int rows = job.at("rows"), vocab = job.at("vocab");
    require(rows > 0 && rows <= 65536 && vocab > 0 && vocab <= 1048576, "invalid logit dimensions");
    const std::string a = job.at("a"), b = job.at("b");
    require(std::filesystem::file_size(a) == uint64_t(rows) * vocab * 4 && std::filesystem::file_size(b) == uint64_t(rows) * vocab * 4, "truncated or trailing logits");
    std::ifstream left(a, std::ios::binary), right(b, std::ios::binary);
    std::vector<float> x(vocab), y(vocab);
    json positions = json::array();
    double ab = 0, ba = 0, maximum = 0;
    int agreed = 0;
    for (int i = 0; i < rows; ++i) {
        require(bool(left.read(reinterpret_cast<char *>(x.data()), vocab * 4)) && bool(right.read(reinterpret_cast<char *>(y.data()), vocab * 4)), "logit read failed");
        auto row = metrics(x, y);
        row["index"] = i;
        ab += row["a_to_b_kl_nats"].get<double>();
        ba += row["b_to_a_kl_nats"].get<double>();
        maximum = std::max(maximum, row["max_abs_logit_difference"].get<double>());
        agreed += row["top1_agreed"].get<bool>();
        positions.push_back(row);
    }
    write_json(job.at("output"), {{"positions", positions}, {"a_to_b_kl_nats", ab / rows}, {"b_to_a_kl_nats", ba / rows},
               {"top1_agreed", agreed}, {"top1_agreement", double(agreed) / rows}, {"max_abs_logit_difference", maximum}});
}

static void prepare(llama_model * model, const json & job, const std::string & hash) {
    const auto vocab = llama_model_get_vocab(model);
    const int depth = job.at("depth"), rows = job.at("rows");
    const bool final_window = job.value("window", std::string("final")) == "final";
    require(depth >= 64 && rows > 0 && rows < depth && depth + rows + 512 <= 1048576, "invalid depth or rows");
    std::ifstream source(job.at("text").get<std::string>());
    require(source.good(), "missing source text");
    auto body = tokenize(vocab, {std::istreambuf_iterator<char>(source), {}});
    auto tokens = tokenize(vocab, "<|im_start|>user\nRead this document and remember its secret records.\n");
    const size_t begin = tokens.size();
    const size_t rotation = job.at("seed").get<uint64_t>() % body.size();
    const std::string needle_mode = job.value("needle_mode", std::string("sequential"));
    require(needle_mode == "sequential" || needle_mode == "independent", "invalid needle mode");
    const int total = final_window ? depth : depth + rows;
    const int split = final_window ? depth - rows : depth;
    for (int i = tokens.size(); i < std::max(total, depth + 64); ++i) { tokens.push_back(body[(rotation + i - begin) % body.size()]); }
    json needles = json::array();
    const auto entries = job.at("needles");
    size_t previous_end = begin;
    for (size_t i = 0; i < entries.size(); ++i) {
        const std::string key = entries[i].at("key"), answer = entries[i].at("answer");
        auto text = tokenize(vocab, "\nSecret record " + key + " has value " + answer + ".\n");
        const size_t position = begin + (depth - begin) * (i + 1) / (entries.size() + 1);
        require(position >= previous_end && position + text.size() < size_t(depth), "needles overlap or exceed document");
        std::copy(text.begin(), text.end(), tokens.begin() + position);
        auto query = tokenize(vocab, "\nWhat is the value of secret record " + key + "? Reply with only its value.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n");
        auto next_query = tokenize(vocab, "<|im_end|>\n<|im_start|>user\nWhat is the value of secret record " + key + "? Reply with only its value.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n");
        needles.push_back({{"query_next", next_query}, {"key", key}, {"expected", answer}, {"position", position}, {"inserted_tokens", text}, {"query", query}});
        previous_end = position + text.size();
    }
    const std::vector<llama_token> tail(tokens.begin() + split, tokens.begin() + total);
    const std::vector<llama_token> probe(tokens.begin() + depth, tokens.begin() + depth + 64);
    tokens.resize(split);
    auto settings = job.at("settings");
    size_t reserve = 512;
    if (needle_mode == "sequential") {
        size_t query_tokens = 0;
        for (const auto & needle : needles) { query_tokens += std::max(needle.at("query").size(), needle.at("query_next").size()) + 32; }
        reserve = std::max(reserve, query_tokens + 64);
    }
    const size_t capacity = std::max<size_t>(total + 512, depth + reserve);
    require(capacity + 255 <= 1048576, "sequential queries exceed context capacity");
    settings["n_ctx"] = (capacity + 255) / 256 * 256;
    write_json(job.at("output"), {{"version", 2}, {"needle_mode", needle_mode}, {"probe_tokens", probe}, {"context_depth", depth}, {"window", final_window ? "final" : "after"}, {"model_sha256", hash}, {"settings", settings}, {"tokens", tokens}, {"tail", tail}, {"needles", needles}, {"source", job.at("source")}, {"seed", job.at("seed")}});
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: llama-hybrid-depth JOB.json");
        const auto job = read_json(argv[1]);
        const std::string mode = job.at("mode");
        if (mode == "compare") { compare(job); return 0; }
        const std::string model_path = job.at("model");
        const auto hash = hybrid::file_hash(model_path);
        require(hash == job.at("model_sha256"), "model hash mismatch");
        const int gpu = job.at("gpu_layers");
        require(gpu >= 0 && gpu <= 999, "invalid GPU layers");
        llama_backend_init();
        require(!gpu || llama_supports_gpu_offload(), "GPU unavailable");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = mode == "prepare" ? 0 : gpu;
        mp.vocab_only = mode == "prepare";
        hybrid::model_ptr model(llama_model_load_from_file(model_path.c_str(), mp));
        require(bool(model), "model load failed");
        if (mode == "prepare") { prepare(model.get(), job, hash); return 0; }
        require(mode == "export" || mode == "evaluate" || mode == "probe", "unknown mode");
        require(!gpu || llama_model_n_devices(model.get()) > 0, "no GPU model device");
        auto fixture = read_json(job.at("fixture"));
        require(fixture.at("model_sha256") == hash, "fixture model mismatch");
        require(fixture.at("version") == 2, "regenerate depth fixture v2");
        const auto identity = binding(fixture);
        const bool final_window = fixture.value("window", std::string("after")) == "final";
        require(same_json(job.at("binding"), identity), "document or YaRN mismatch");
        auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
        const auto tail = fixture.at("tail").get<std::vector<llama_token>>();
        const auto probe = fixture.at("probe_tokens").get<std::vector<llama_token>>();
        require(probe.size() == 64, "exactly 64 handoff probe tokens required");
        const auto settings = fixture.at("settings");
        const int n_ctx = settings.at("n_ctx"), batch_size = job.at("batch_size"), vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(n_ctx > 0 && n_ctx <= 1048576 && !tokens.empty() && !tail.empty() && tokens.size() + tail.size() + 512 <= size_t(n_ctx), "invalid context bounds");
        require(batch_size > 0 && batch_size <= 8192, "invalid prefill batch");
        for (auto sequence : {tokens, tail, probe}) {
            for (auto t : sequence) { require(t >= 0 && t < vocab, "invalid token"); }
        }
        for (const auto & needle : fixture.at("needles")) {
            require(needle.at("query").size() < 256, "query too long");
            for (const auto * key : {"query", "query_next"}) {
                require(needle.at(key).size() < 256, "query too long");
                for (auto t : needle.at(key).get<std::vector<llama_token>>()) { require(t >= 0 && t < vocab, "invalid query token"); }
            }
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = cp.n_ubatch = std::max(batch_size, 32);
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = cp.op_offload = gpu > 0;
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        const float factor = settings.at("yarn_factor");
        require(std::isfinite(factor) && factor >= 1 && factor <= 4, "invalid YaRN factor");
        cp.rope_freq_scale = 1.0f / factor;
        cp.rope_freq_base = settings.at("rope_theta");
        cp.yarn_orig_ctx = settings.at("original_context");
        require(settings.at("kv_type") == "q8_0" && settings.at("recurrent_type") == "f32" && settings.at("yarn_ext_factor") == 1 &&
                settings.at("yarn_attn_factor") == 1 && settings.at("yarn_beta_fast") == 32 && settings.at("yarn_beta_slow") == 1 &&
                cp.rope_freq_base == 10000000 && cp.yarn_orig_ctx == 262144, "unsupported YaRN or state settings");
        cp.yarn_ext_factor = 1;
        cp.yarn_attn_factor = 1;
        cp.yarn_beta_fast = 32;
        cp.yarn_beta_slow = 1;
        hybrid::context_ptr ctx(llama_init_from_model(model.get(), cp));
        require(bool(ctx), "context load failed");
        const std::string state = job.at("state");
        if (mode == "probe") {
            if (final_window) { tokens.insert(tokens.end(), tail.begin(), tail.end()); }
            load_state(ctx.get(), state, tokens, identity, true);
            const size_t depth = tokens.size();
            tokens.insert(tokens.end(), probe.begin(), probe.end());
            const std::string path = job.at("logits");
            write_logits(ctx.get(), tokens, depth, 1, vocab, path);
            write_json(job.at("output"), {{"binding", identity}, {"vocab", vocab}, {"rows", 64}, {"depth", depth},
                       {"logits_sha256", hybrid::file_hash(path)}, {"probe_tokens", probe}});
            return 0;
        }
        if (mode == "export") {
            export_transfer(ctx.get(), state, tokens, batch_size, job.value("chunk_tokens", 8192), identity);
            return 0;
        }
        json transfer = nullptr;
        if (job.value("import", false)) { transfer = import_transfer(ctx.get(), state, tokens, identity, vocab, !final_window); }
        else {
            decode(ctx.get(), tokens, 0, tokens.size(), batch_size);
            if (!final_window) { save_state(ctx.get(), state, tokens, identity); }
        }
        const std::string logits_path = job.at("logits");
        auto stream = tokens;
        stream.insert(stream.end(), tail.begin(), tail.end());
        write_logits(ctx.get(), stream, tokens.size(), 32, vocab, logits_path);
        const auto & checkpoint = final_window ? stream : tokens;
        if (final_window) { save_state(ctx.get(), state, checkpoint, identity); }
        size_t restores = 0;
        if (!final_window) { load_state(ctx.get(), state, checkpoint, identity, false); ++restores; }
        auto reference = checkpoint;
        reference.insert(reference.end(), probe.begin(), probe.end());
        const std::string reference_path = job.at("reference_logits");
        write_logits(ctx.get(), reference, checkpoint.size(), 1, vocab, reference_path);
        json answers = json::array();
        const bool sequential = fixture.at("needle_mode") == "sequential";
        auto query = checkpoint;
        for (const auto & needle : fixture.at("needles")) {
            // Restore once for a sequential chat. Independent probes restore before every question.
            if (!sequential || answers.empty()) {
                load_state(ctx.get(), state, checkpoint, identity, false);
                ++restores;
                query = checkpoint;
            }
            const size_t begin = query.size();
            const auto suffix = needle.at(sequential && !answers.empty() ? "query_next" : "query").get<std::vector<llama_token>>();
            query.insert(query.end(), suffix.begin(), suffix.end());
            require(query.size() + 32 <= llama_n_ctx(ctx.get()), "needle chain exceeds context capacity");
            decode(ctx.get(), query, begin, query.size(), 32);
            std::vector<llama_token> generated;
            for (int i = 0; i < 32; ++i) {
                const auto token = hybrid::greedy(ctx.get(), model.get());
                if (llama_vocab_is_eog(llama_model_get_vocab(model.get()), token)) { break; }
                generated.push_back(token);
                query.push_back(token);
                decode(ctx.get(), query, query.size() - 1, query.size(), 1);
            }
            int n = llama_detokenize(llama_model_get_vocab(model.get()), generated.data(), generated.size(), nullptr, 0, true, false);
            require(n <= 0, "detokenization sizing failed");
            std::string answer(-n, '\0');
            if (n) { require(llama_detokenize(llama_model_get_vocab(model.get()), generated.data(), generated.size(), &answer[0], answer.size(), true, false) == -n, "detokenization failed"); }
            answers.push_back({{"key", needle.at("key")}, {"expected", needle.at("expected")}, {"text", answer}, {"tokens", generated}});
        }
        write_json(job.at("output"), {{"binding", identity}, {"vocab", vocab}, {"rows", tail.size()}, {"batch_size", batch_size}, {"gpu_layers", gpu},
                   {"logits_sha256", hybrid::file_hash(logits_path)}, {"state", read_json(state + ".json")}, {"imported_transfer", transfer}, {"answers", answers},
                   {"logit_positions", "after each forced token; first at prefix token count"}, {"snapshot_depth", checkpoint.size()}, {"state_restores", restores}, {"needle_mode", fixture.at("needle_mode")},
                   {"reference_logits_sha256", hybrid::file_hash(reference_path)}, {"reference_rows", 64}, {"probe_tokens", probe}});
    } catch (const std::exception & error) {
        std::cerr << "depth refused: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
