#include "hybrid-state.h"
#include "llama-ext.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using hybrid::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::string read_bytes(std::istream & in, size_t n) {
    std::string data(n, '\0');
    require(bool(in.read(&data[0], n)), "truncated reference file");
    return data;
}

static uint32_t u32(const std::string & data, size_t offset = 0) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) { value |= uint32_t(uint8_t(data.at(offset + i))) << (8 * i); }
    return value;
}

static std::string little32(uint32_t value) {
    std::string data(4, '\0');
    for (int i = 0; i < 4; ++i) { data[i] = char(value >> (8 * i)); }
    return data;
}

static std::string digest(const std::string & data) {
    unsigned char hash[32];
    sha256_hash(hash, reinterpret_cast<const unsigned char *>(data.data()), data.size());
    return {reinterpret_cast<const char *>(hash), sizeof(hash)};
}

static bool valid_hash(const std::string & hash) {
    return hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos;
}

static std::vector<double> normalize(const std::vector<double> & logits) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (double value : logits) {
        require(std::isfinite(value), "nonfinite log-probability");
        sum += std::exp(value - maximum);
    }
    const double z = maximum + std::log(sum);
    auto result = logits;
    for (auto & value : result) { value -= z; }
    return result;
}

static std::vector<double> log_probs(const float * logits, int vocab) {
    require(logits, "missing score logits");
    return normalize({logits, logits + vocab});
}

static double kl(const std::vector<double> & p, const std::vector<double> & q) {
    double result = 0;
    for (size_t i = 0; i < p.size(); ++i) { result += std::exp(p[i]) * (p[i] - q[i]); }
    require(std::isfinite(result), "nonfinite KL");
    return std::max(0.0, result);
}

static size_t top1(const std::vector<double> & p) {
    return std::max_element(p.begin(), p.end()) - p.begin();
}

static json spread(std::vector<double> values) {
    require(!values.empty(), "empty score summary");
    std::sort(values.begin(), values.end());
    auto percentile = [&](double fraction) {
        const double index = fraction * (values.size() - 1);
        const size_t lower = size_t(index), upper = std::min(lower + 1, values.size() - 1);
        return values[lower] + (values[upper] - values[lower]) * (index - lower);
    };
    return {{"mean", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
            {"median", percentile(.5)}, {"p90", percentile(.9)}, {"p99", percentile(.99)}, {"max", values.back()}};
}

static std::string encode(const std::vector<double> & p) {
    std::string row;
    row.reserve(2 * p.size());
    for (double value : p) {
        const auto half = ggml_fp32_to_fp16(float(value));
        require(std::isfinite(ggml_fp16_to_fp32(half)), "log-probability exceeds float16 range");
        row += char(half & 255);
        row += char(half >> 8);
    }
    return row;
}

static std::vector<double> decode(const std::string & row, int vocab) {
    std::vector<double> p(vocab);
    double mass = 0;
    for (int i = 0; i < vocab; ++i) {
        p[i] = ggml_fp16_to_fp32(ggml_fp16_t(uint8_t(row[12 + 2 * i]) | uint16_t(uint8_t(row[13 + 2 * i])) << 8));
        require(std::isfinite(p[i]), "nonfinite reference log-probability");
        require(p[i] <= 0, "positive reference log-probability");
        mass += std::exp(p[i]);
    }
    require(std::abs(mass - 1) <= .02, "reference probability mass mismatch");
    return normalize(p);
}

struct reference_reader {
    std::ifstream in;
    json header;
    std::streampos payload;
    size_t n, m;
    int vocab;
    size_t original_top1 = 0;
    std::vector<llama_token> continuation;

    reference_reader(const std::string & path, const json & fixture, const std::vector<llama_token> & tokens,
                     const std::vector<llama_token> & next) : in(path, std::ios::binary), n(tokens.size()), m(next.size()), continuation(next) {
        require(in.good(), "cannot read reference file");
        require(read_bytes(in, 8) == "HYBLOG01", "invalid reference magic");
        const auto size = u32(read_bytes(in, 4));
        require(size > 0 && size <= 8 * 1024 * 1024, "reference header exceeds limit");
        header = hybrid::parse_json(read_bytes(in, size), 8 * 1024 * 1024);
        require(header.at("format") == "f16-logsoftmax-le-v1", "unsupported reference format");
        const auto expected = fixture.at("reference_model_sha256").get<std::string>();
        require(valid_hash(expected) && header.at("model_sha256") == expected, "reference model hash mismatch");
        require(header.at("n") == n && header.at("m") == m, "reference N/M mismatch");
        std::vector<llama_token> all = tokens;
        all.insert(all.end(), next.begin(), next.end());
        require(header.at("token_sha256") == hybrid::prefix_hash(all) && header.at("tokens") == tokens &&
                header.at("continuation") == next, "reference token mismatch");
        const int64_t v = header.at("vocab");
        require(v > 0 && v <= 1048576, "invalid reference vocabulary");
        vocab = int(v);
        payload = in.tellg();
        in.seekg(0, std::ios::end);
        require(in.tellg() - payload == std::streamoff(m * (44 + 2 * uint64_t(vocab))), "truncated or trailing reference file");
        in.seekg(payload);
        for (size_t i = 0; i < m; ++i) { read(i); }
        in.seekg(payload);
    }

    std::vector<double> read(size_t i) {
        const auto row = read_bytes(in, 12 + 2 * vocab);
        require(u32(row) == n + i, "reference position mismatch");
        require(u32(row, 4) == uint32_t(continuation[i]), "reference target mismatch");
        original_top1 = u32(row, 8);
        require(original_top1 < size_t(vocab), "invalid reference top1");
        const auto checksum = read_bytes(in, 32);
        require(checksum == digest(row), "reference row checksum mismatch");
        return decode(row, vocab);
    }
};

struct pending_file {
    std::string path;
    std::ofstream out;
    explicit pending_file(const std::string & destination) : path(destination + ".tmp") {
        require(!std::filesystem::exists(destination) && !std::filesystem::exists(path), "output already exists");
        out.open(path, std::ios::binary);
        require(out.good(), "cannot open score output");
        std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    }
    ~pending_file() { if (!path.empty()) { std::error_code error; std::filesystem::remove(path, error); } }
    void publish(const std::string & destination) {
        out.close();
        require(out.good(), "cannot write score output");
        std::filesystem::rename(path, destination);
        path.clear();
    }
};

static hybrid::context_ptr context(llama_model * model, uint32_t depth, bool gpu) {
    auto cp = llama_context_default_params();
    cp.n_ctx = depth;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = cp.op_offload = gpu;
    hybrid::context_ptr result(llama_init_from_model(model, cp));
    require(bool(result), "cannot create score context");
    return result;
}

static void self_test() {
    // An asymmetric analytic pair fixes KL direction; percentiles use linear interpolation.
    const std::vector<double> p = {std::log(.8), std::log(.2)}, q = {std::log(.6), std::log(.4)};
    require(std::abs(kl(p, q) - .0915162218494358) < 1e-12 && kl(p, p) == 0, "KL direction control failed");
    const auto stats = spread({0, 1, 2, 3, 4});
    require(stats.at("mean") == 2 && stats.at("median") == 2 && std::abs(stats.at("p90").get<double>() - 3.6) < 1e-12 &&
            std::abs(stats.at("p99").get<double>() - 3.96) < 1e-12 && stats.at("max") == 4, "percentile control failed");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        require(argc == 5, "usage: llama-hybrid-score MODEL FIXTURE.json OUTPUT.json GPU_LAYERS");
        std::ifstream input(argv[2], std::ios::binary);
        require(input.good(), "cannot read score fixture");
        input.seekg(0, std::ios::end);
        require(input.tellg() > 0 && input.tellg() <= 8 * 1024 * 1024, "score fixture exceeds limit");
        input.seekg(0);
        const auto fixture = hybrid::parse_json({std::istreambuf_iterator<char>(input), {}}, 8 * 1024 * 1024);
        for (const auto & name : {"tokens", "continuation"}) {
            require(fixture.at(name).is_array(), "score tokens must be arrays");
            for (const auto & token : fixture.at(name)) {
                require(token.is_number_integer() && token.get<int64_t>() >= 0 && token.get<int64_t>() <= INT32_MAX, "invalid score token");
            }
        }
        const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
        const auto next = fixture.at("continuation").get<std::vector<llama_token>>();
        const size_t n = tokens.size(), m = next.size();
        const int64_t requested_batch = fixture.value("score_batch", int64_t(32));
        const int64_t max_ctx = fixture.value("max_ctx", int64_t(16384));
        require(requested_batch > 0 && requested_batch <= 256 && max_ctx > 0 && max_ctx <= 1048576, "score context or batch exceeds bounds");
        const int batch_size = int(requested_batch);
        require(n >= 2 && m > 0 && n + m <= 1048576 && n + m <= uint64_t(max_ctx) &&
                batch_size > 0 && batch_size <= 256, "score N/M or batch exceeds bounds");
        const bool writing = fixture.contains("write_reference");
        require(writing != fixture.contains("reference_logits"), "choose reference input or output");
        std::unique_ptr<reference_reader> reference;
        if (!writing) { reference.reset(new reference_reader(fixture.at("reference_logits"), fixture, tokens, next)); }
        const int gpu_layers = std::stoi(argv[4]);
        require(gpu_layers >= 0 && gpu_layers <= 999, "invalid score GPU layers");
        const auto model_hash = hybrid::file_hash(argv[1]);
        require(fixture.at("model_sha256") == model_hash, "score model mismatch");
        require(!std::filesystem::exists(argv[3]) && !std::filesystem::exists(std::string(argv[3]) + ".tmp"), "output already exists");
        llama_backend_init();
        require(gpu_layers == 0 || llama_supports_gpu_offload(), "score GPU unavailable");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "cannot load score model");
        require(gpu_layers == 0 || llama_model_n_devices(model.get()) > 0, "score model has no GPU device");
        const auto layout = hybrid::model_layout(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (reference) {
            require(reference->vocab == vocab && reference->header.at("vocab_sha256") == layout.at("vocab_sha256"), "reference vocabulary mismatch");
            require(reference->header.at("layout_fingerprint") == hybrid::layout_fingerprint(layout), "reference layout mismatch");
        }
        for (const auto & sequence : {tokens, next}) {
            for (auto token : sequence) { require(token >= 0 && token < vocab, "invalid score token"); }
        }
        const auto paths = fixture.value("deltas", std::vector<std::string>{});
        require(!writing || paths.empty(), "reference production requires local prefill");
        const auto source_hash = fixture.value("node_model_sha256", model_hash);
        if (source_hash != model_hash) {
            require(fixture.value("hetero_quant", false) && fixture.at("node_layout_fingerprint") == hybrid::layout_fingerprint(layout), "heterogeneous layout mismatch");
        }
        const uint32_t depth = uint32_t((n + m + 255) / 256 * 256);
        auto local = context(model.get(), depth, gpu_layers > 0);
        hybrid::context_ptr imported;
        if (!paths.empty()) {
            imported = context(model.get(), depth, gpu_layers > 0);
            int64_t end = 0;
            for (const auto & path : paths) {
                const auto size = std::filesystem::file_size(path);
                require(size <= 513 * 1024 * 1024 + 52, "score delta exceeds limit");
                std::ifstream delta_input(path, std::ios::binary);
                const auto delta = hybrid::read_delta(read_bytes(delta_input, size), source_hash, end);
                const int64_t n1 = delta.header.at("n1");
                require(n1 <= int64_t(n - 1), "score delta endpoint mismatch");
                require(delta.header.at("tokens") == std::vector<llama_token>(tokens.begin() + end, tokens.begin() + n1) &&
                        delta.header.at("prefix_hash_n1") == hybrid::prefix_hash({tokens.begin(), tokens.begin() + n1}), "score delta tokens mismatch");
                require(llama_state_seq_set_data_append(imported.get(), reinterpret_cast<const uint8_t *>(delta.state.data()), delta.state.size(), 0, end, n1) == delta.state.size(), "score delta import failed");
                end = n1;
            }
            require(end == int64_t(n - 1), "incomplete score delta chain");
            hybrid::prefill(imported.get(), {tokens.back()}, n - 1);
        }
        hybrid::prefill(local.get(), tokens, 0);
        std::vector<llama_token> all = tokens;
        all.insert(all.end(), next.begin(), next.end());
        json header = {{"format", "f16-logsoftmax-le-v1"}, {"model_sha256", model_hash},
            {"token_sha256", hybrid::prefix_hash(all)}, {"n", n}, {"m", m}, {"vocab", vocab},
            {"tokens", tokens}, {"continuation", next}, {"vocab_sha256", layout.at("vocab_sha256")},
            {"layout_fingerprint", hybrid::layout_fingerprint(layout)}, {"gpu_layers", gpu_layers},
            {"score_batch", batch_size}, {"kv_type", "q8_0"}, {"threads", 2}};
        std::unique_ptr<pending_file> target;
        std::string destination;
        if (writing) {
            destination = fixture.at("write_reference");
            require(destination != argv[3], "reference and result paths overlap");
            target.reset(new pending_file(destination));
            const auto text = header.dump();
            require(text.size() <= 8 * 1024 * 1024, "reference header exceeds limit");
            target->out << "HYBLOG01" << little32(text.size()) << text;
        }
        std::vector<double> local_kls, hybrid_kls, own_kls, encoding_kls;
        int local_agreed = 0, hybrid_agreed = 0, own_agreed = 0, encoding_agreed = 0;
        double encoding_error = 0;
        auto positions = json::array();
        auto emit = [&](size_t i, int index) {
            const auto p = log_probs(llama_get_logits_ith(local.get(), index), vocab);
            const auto arm = imported ? log_probs(llama_get_logits_ith(imported.get(), index), vocab) : p;
            auto row = little32(n + i) + little32(next[i]) + little32(top1(p)) + encode(p);
            const auto encoded = decode(row, vocab);
            encoding_kls.push_back(kl(p, encoded));
            encoding_agreed += top1(p) == top1(encoded);
            for (int v = 0; v < vocab; ++v) {
                const auto half = ggml_fp16_t(uint8_t(row[12 + 2 * v]) | uint16_t(uint8_t(row[13 + 2 * v])) << 8);
                encoding_error = std::max(encoding_error, std::abs(p[v] - ggml_fp16_to_fp32(half)));
            }
            if (target) { target->out << row << digest(row); }
            const auto d = reference ? reference->read(i) : p;
            const size_t dtop1 = reference ? reference->original_top1 : top1(p);
            const double a = kl(d, p), b = kl(d, arm), own = kl(p, arm);
            local_kls.push_back(a);
            hybrid_kls.push_back(b);
            own_kls.push_back(own);
            local_agreed += dtop1 == top1(p);
            hybrid_agreed += dtop1 == top1(arm);
            own_agreed += top1(p) == top1(arm);
            positions.push_back({{"position", n + i},
                {"local_kl_nats", a}, {"local_top1_agreed", dtop1 == top1(p)},
                {"hybrid_kl_nats", imported ? json(b) : json(nullptr)},
                {"hybrid_top1_agreed", imported ? json(dtop1 == top1(arm)) : json(nullptr)},
                {"local_to_hybrid_kl_nats", imported ? json(own) : json(nullptr)},
                {"local_to_hybrid_top1_agreed", imported ? json(top1(p) == top1(arm)) : json(nullptr)}});
        };
        emit(0, -1);
        for (size_t offset = 0; offset + 1 < m;) {
            const size_t count = std::min<size_t>(batch_size, m - 1 - offset);
            llama_batch batch = llama_batch_init(count, 0, 1);
            for (size_t j = 0; j < count; ++j) {
                batch.token[j] = next[offset + j];
                batch.pos[j] = n + offset + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = true;
            }
            batch.n_tokens = count;
            const int local_code = llama_decode(local.get(), batch);
            const int imported_code = imported ? llama_decode(imported.get(), batch) : 0;
            llama_batch_free(batch);
            require(local_code == 0 && imported_code == 0, "teacher-forced decode failed");
            for (size_t j = 0; j < count; ++j) { emit(offset + j + 1, j); }
            offset += count;
        }
        auto summary = [&](const std::vector<double> & values, int agreed) {
            return json{{"kl_nats", spread(values)}, {"top1_agreed", agreed}, {"top1_agreement", double(agreed) / m}};
        };
        auto identity = header;
        identity.erase("tokens");
        identity.erase("continuation");
        json result = {{"stage", "reference_scoring"}, {"status", "measured"}, {"identity", identity},
            {"reference_model_sha256", reference ? reference->header.at("model_sha256") : json(model_hash)},
            {"node_model_sha256", source_hash}, {"positions", positions}, {"local", summary(local_kls, local_agreed)},
            {"hybrid", imported ? summary(hybrid_kls, hybrid_agreed) : json(nullptr)},
            {"local_to_hybrid", imported ? summary(own_kls, own_agreed) : json(nullptr)},
            {"encoding", {{"kl_nats", spread(encoding_kls)}, {"max_abs_log_probability_error", encoding_error},
                          {"top1_agreed", encoding_agreed}, {"bytes_per_position", 2 * vocab + 44}}}};
        pending_file output(argv[3]);
        output.out << result.dump(2) << '\n';
        if (target) { target->publish(destination); }
        output.publish(argv[3]);
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "score failed: " << error.what() << '\n';
        return 2;
    }
}
