#include "hybrid-state.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct comparison {
    double kl;
    bool top1;
};

static comparison compare(const float * p, const float * q, size_t n) {
    require(n > 0, "empty distribution");
    double pmax = -std::numeric_limits<double>::infinity();
    double qmax = pmax;
    size_t ptop = 0, qtop = 0;
    for (size_t i = 0; i < n; ++i) {
        require(std::isfinite(p[i]) && std::isfinite(q[i]), "nonfinite logit");
        if (p[i] > pmax) { pmax = p[i]; ptop = i; }
        if (q[i] > qmax) { qmax = q[i]; qtop = i; }
    }
    double psum = 0, qsum = 0;
    for (size_t i = 0; i < n; ++i) {
        psum += std::exp(p[i] - pmax);
        qsum += std::exp(q[i] - qmax);
    }
    const double plogz = pmax + std::log(psum);
    const double qlogz = qmax + std::log(qsum);
    double kl = 0;
    for (size_t i = 0; i < n; ++i) {
        kl += std::exp(p[i] - plogz) * ((p[i] - plogz) - (q[i] - qlogz));
    }
    return {std::max(0.0, kl), ptop == qtop};
}

static bool within_bounds(double kl, double top1) {
    return std::isfinite(kl) && kl >= 0 && kl <= 0.01 && top1 >= 0.95 && top1 <= 1;
}

static void controls() {
    // The metric and gate must detect a known distribution violation, with an identical positive control.
    const float p[] = {2, -2}, q[] = {-2, 2};
    auto same = compare(p, p, 2);
    auto different = compare(p, q, 2);
    require(same.kl < 1e-12 && same.top1 && within_bounds(same.kl, 1), "identical distribution control failed");
    require(different.kl > 0.01 && !different.top1 && !within_bounds(different.kl, 0), "distribution violation control survived");
    const float near[] = {1, 0}, tied_p[] = {0.001f, -0.001f}, tied_q[] = {-0.001f, 0.001f};
    auto kl_only = compare(p, near, 2);
    auto top1_only = compare(tied_p, tied_q, 2);
    require(kl_only.top1 && !within_bounds(kl_only.kl, 1), "KL-only gate control survived");
    require(top1_only.kl < 0.01 && !top1_only.top1 && !within_bounds(top1_only.kl, 0), "top1-only gate control survived");
    require(within_bounds(0.01, 0.95) && !within_bounds(0.010001, 1) && !within_bounds(0, 0.949), "gate boundary control failed");
    std::cout << "PASS distribution controls: identity and boundaries accepted; KL-only and top1-only violations rejected\n";
}

static llama_token sample(llama_context * ctx, const llama_vocab * vocab, bool ignore_eos) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "no sample logits");
    llama_token best = -1;
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
        if (ignore_eos && llama_vocab_is_eog(vocab, token)) { continue; }
        if (best == -1 || logits[token] > logits[best]) { best = token; }
    }
    require(best >= 0, "empty sampling vocabulary");
    return best;
}

static std::string read_file(const char * path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open token fixture");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static void run(llama_model * model, const hybrid::json & fixture, const char * path) {
    const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
    const auto continuation = fixture.at("greedy_64").get<std::vector<llama_token>>();
    const bool ignore_eos = fixture.value("ignore_eos", false);
    const auto * vocab = llama_model_get_vocab(model);
    require(!tokens.empty() && tokens.size() <= 8192 && continuation.size() == 64, "invalid teacher-forcing fixture");
    for (auto token : tokens) { require(token >= 0 && token < llama_vocab_n_tokens(vocab), "invalid prompt token"); }
    for (auto token : continuation) { require(token >= 0 && token < llama_vocab_n_tokens(vocab), "invalid continuation token"); }
    auto imported = hybrid::make_context(model, tokens.size() + 256);
    const std::string filename(path);
    const std::string suffix = ".tokens.json";
    require(filename.size() > suffix.size() && filename.substr(filename.size() - suffix.size()) == suffix,
            "teacher fixture must end in .tokens.json");
    const auto stem = filename.substr(0, filename.size() - suffix.size());
    const auto n0 = fixture.value("n0", 0);
    require(n0 >= 0 && (size_t) n0 < tokens.size() - 1, "invalid teacher delta start");
    const auto delta = hybrid::read_delta(read_file((stem + ".delta").c_str()), fixture.at("model_sha256"), n0);
    require(delta.header.at("n1") == tokens.size() - 1, "teacher delta endpoint mismatch");
    require(delta.header.at("tokens").get<std::vector<llama_token>>() ==
            std::vector<llama_token>(tokens.begin() + n0, tokens.end() - 1), "teacher delta token mismatch");
    require(delta.header.at("prefix_hash_n1") == hybrid::prefix_hash({tokens.begin(), tokens.end() - 1}),
            "teacher delta prefix mismatch");
    if (n0 > 0) { hybrid::load_state(imported.get(), read_file((stem + ".prefix.state").c_str())); }
    require(llama_state_seq_set_data_append(imported.get(), reinterpret_cast<const uint8_t *>(delta.state.data()),
            delta.state.size(), 0, n0, tokens.size() - 1) == delta.state.size(), "teacher delta import failed");
    hybrid::prefill(imported.get(), {tokens.back()}, tokens.size() - 1);
    auto split = hybrid::make_context(model, tokens.size() + 256);
    auto whole = hybrid::make_context(model, tokens.size() + 256);
    hybrid::prefill(split.get(), {tokens.begin(), tokens.end() - 1}, 0);
    hybrid::prefill(split.get(), {tokens.back()}, tokens.size() - 1);
    hybrid::prefill(whole.get(), tokens, 0);
    double kl_sum = 0;
    size_t agreed = 0;
    for (size_t i = 0; i < continuation.size(); ++i) {
        require(sample(imported.get(), vocab, ignore_eos) == continuation[i], "imported greedy differs from observed continuation");
        require(sample(split.get(), vocab, ignore_eos) == continuation[i], "same-partition greedy mismatch");
        // Both independent contexts receive the exact same observed continuation, even when whole-prefill top1 differs.
        const auto metric = compare(llama_get_logits_ith(imported.get(), -1), llama_get_logits_ith(whole.get(), -1),
                                    llama_vocab_n_tokens(vocab));
        kl_sum += metric.kl;
        agreed += metric.top1;
        hybrid::prefill(imported.get(), {continuation[i]}, tokens.size() + i);
        hybrid::prefill(split.get(), {continuation[i]}, tokens.size() + i);
        hybrid::prefill(whole.get(), {continuation[i]}, tokens.size() + i);
    }
    const double mean_kl = kl_sum / continuation.size();
    const double agreement = (double) agreed / continuation.size();
    std::cout << "METRICS " << path << " same_partition=64/64 mean_kl=" << mean_kl
              << " top1=" << agreed << "/64 agreement=" << agreement << '\n';
    require(within_bounds(mean_kl, agreement), "teacher-forced KL/top1 bounds exceeded");
    std::cout << "PASS teacher-forced distribution bounds " << path << '\n';
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "usage: llama-hybrid-check-distributions MODEL.gguf TOKENS.json [TOKENS.json ...]");
        controls();
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], params));
        require(bool(model), "cannot load model");
        const auto hash = hybrid::file_hash(argv[1]);
        for (int i = 2; i < argc; ++i) {
            auto fixture = hybrid::parse_json(read_file(argv[i]));
            require(fixture.at("model_sha256") == hash, "teacher fixture model mismatch");
            run(model.get(), fixture, argv[i]);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
