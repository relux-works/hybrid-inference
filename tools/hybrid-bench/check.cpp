#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-io.h"
#include "llama-memory-hybrid.h"
#ifdef LLAMA_HYBRID_SERVER_SNAPSHOTS
#include "server-common.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::string checker_backend_label(const std::string & name, const std::string * expected = nullptr) {
    const std::string label = name == "CUDA" ? "cuda" : (name == "MTL" || name == "Metal") ? "metal" : name == "CPU" ? "cpu" : name;
    require(expected == nullptr || *expected == label, "checker backend label/device mismatch");
    return label;
}

class state_counter : public llama_io_write_i {
    size_t size = 0;
public:
    void write(const void *, size_t n) override { size += n; }
    void write_tensor(ggml_tensor *, size_t, size_t n) override { size += n; }
    size_t n_bytes() override { return size; }
};

static size_t attention_state_size(llama_context * ctx) {
    auto * memory = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    require(memory != nullptr, "state probes require hybrid memory");
    state_counter counter;
    memory->get_mem_attn()->state_write(counter, 0, 0);
    return counter.n_bytes();
}

static std::string read_file(const char * path, size_t limit = 513 * 1024 * 1024 + 52) {
    std::ifstream source(path, std::ios::binary);
    require(source.good(), "cannot read check input");
    source.seekg(0, std::ios::end);
    require(source.tellg() >= 0 && uint64_t(source.tellg()) <= limit, "check input exceeds limit");
    source.seekg(0);
    return {std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
}

static bool snapshot_state_matches(std::istream & source, size_t n, const std::string & state) {
    require(state.size() >= 8 && n <= (std::numeric_limits<size_t>::max() - 12) / 4, "invalid state probe native state");
    const size_t offset = 12 + 4 * n;
    source.seekg(0, std::ios::end);
    const auto size = source.tellg();
    require(size >= 0 && uint64_t(size) >= offset, "invalid state probe snapshot size");
    if (uint64_t(size) - offset != state.size() - 8) { return false; }
    source.seekg(offset, std::ios::beg);
    char buffer[65536];
    for (size_t at = 8; at < state.size();) {
        const size_t count = std::min(sizeof(buffer), state.size() - at);
        source.read(buffer, count);
        require(source.gcount() == std::streamsize(count), "cannot read state probe snapshot");
        if (std::memcmp(buffer, state.data() + at, count) != 0) { return false; }
        at += count;
    }
    return true;
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

static bool within_session_band(double kl, double mean, int agreed, int k, int divergence) {
    return kl <= mean && agreed >= k - 1 && divergence == -1;
}

// Exact replay and component identities gate append import; L-H KL is a measured cost.
static hybrid::json append_replay_controls(const hybrid::json & result) {
    hybrid::json controls = hybrid::json::object();
    auto exact = [](const hybrid::json & row) {
        return row.at("exact").get<bool>() && row.at("max_abs_logit_diff") == 0 && row.at("mean_kl_nats") == 0 &&
            row.at("top1_agreed") == 64 && row.at("free_running_first_divergence") == -1;
    };
    const auto & probes = result.at("probes");
    for (const auto & pair : std::vector<std::pair<std::string, std::string>>{
            {"NODE", "B"}, {"RESTORE", "B"}, {"BSTATE", "B"}, {"NODE_RANGE", "B"},
            {"OWN_NATIVE", "A"}, {"OWN_RANGE", "A"}, {"CHAIN", "H"},
            {"CHAIN_ATTENTION", "B"}, {"CHAIN_RECURRENT", "H"}}) {
        const auto & probe = probes.at(pair.first);
        bool equal = exact(probe.at(pair.second));
        const auto & states = probe.at("roundtrips");
        if (pair.first != "NODE" && pair.first != "CHAIN") {
            equal &= !states.empty();
            for (const auto & state : states) {
                if (pair.first == "RESTORE" || pair.first == "BSTATE") {
                    equal &= state.contains("bytes_equal") ? state.at("bytes_equal").get<bool>() : state.at("snapshot_bytes_equal").get<bool>();
                } else if (pair.first == "CHAIN_ATTENTION" || pair.first == "CHAIN_RECURRENT") {
                    equal &= state.at("recurrent_equal").get<bool>();
                } else {
                    equal &= state.at("bytes_equal").get<bool>();
                }
            }
        }
        controls[pair.first + "-" + pair.second] = equal;
    }
    controls["A-L"] = exact(result.at("factorial").at("A-L"));
    return controls;
}

static bool append_tokens_agree(int agreed) {
    return agreed >= 63;
}

static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static hybrid::context_ptr context(llama_model * model, int depth, bool gpu, int threads = 2) {
    auto cp = llama_context_default_params();
    cp.n_ctx = depth;
    cp.n_batch = cp.n_ubatch = 256;
    cp.swa_full = false;
    cp.n_threads = cp.n_threads_batch = threads;
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

struct checkpoint_observation {
    std::vector<float> logits;
    std::vector<llama_token> teacher;
    std::vector<llama_token> free_tokens;
};

static checkpoint_observation observe(llama_context * ctx, llama_model * model, size_t depth, int k,
                                      const std::vector<llama_token> & teacher = {}) {
    checkpoint_observation result;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto state = hybrid::save_state(ctx);
    const auto first = hybrid::greedy(ctx, model);
    for (int i = 0; i < k; ++i) {
        const auto * logits = llama_get_logits_ith(ctx, -1);
        result.logits.insert(result.logits.end(), logits, logits + vocab);
        const auto next = teacher.empty() ? hybrid::greedy(ctx, model) : teacher[i];
        result.teacher.push_back(next);
        hybrid::prefill(ctx, {next}, depth + i);
    }
    hybrid::load_state(ctx, state);
    for (int i = 0; i < k; ++i) {
        const auto next = i == 0 ? first : hybrid::greedy(ctx, model);
        result.free_tokens.push_back(next);
        hybrid::prefill(ctx, {next}, depth + i);
    }
    return result;
}

static hybrid::json score_observations(const checkpoint_observation & p, const checkpoint_observation & q, size_t depth, int k) {
    const size_t vocab = p.logits.size() / k;
    require(p.logits.size() == q.logits.size() && vocab > 0, "checkpoint logits shape mismatch");
    hybrid::json result = {{"positions", hybrid::json::array()}, {"free_running_first_divergence", -1}};
    double sum = 0, difference = 0;
    int agreed = 0;
    bool exact = true;
    std::vector<double> values;
    for (int i = 0; i < k; ++i) {
        const auto * l = p.logits.data() + i * vocab;
        const auto * h = q.logits.data() + i * vocab;
        const bool equal = std::memcmp(l, h, vocab * sizeof(float)) == 0;
        const double value = equal ? 0 : compare(l, h, vocab);
        const double diff = max_difference(l, h, vocab);
        const bool top1 = std::max_element(l, l + vocab) - l == std::max_element(h, h + vocab) - h;
        result["positions"].push_back({{"position", depth + i}, {"exact", equal}, {"top1_agreed", top1},
            {"kl_nats", value}, {"max_abs_logit_diff", diff}});
        sum += value;
        difference = std::max(difference, diff);
        agreed += top1;
        exact &= equal;
        values.push_back(value);
        if (p.free_tokens[i] != q.free_tokens[i] && result["free_running_first_divergence"] == -1) {
            result["free_running_first_divergence"] = i;
        }
    }
    std::sort(values.begin(), values.end());
    result.update({{"mean_kl_nats", sum / k}, {"p95_kl_nats", values[size_t(std::ceil(.95 * k)) - 1]},
        {"max_abs_logit_diff", difference}, {"top1_agreed", agreed}, {"exact", exact}});
    return result;
}

static void replay_schedule(llama_context * ctx, const std::vector<llama_token> & tokens, const hybrid::json & schedule, size_t start = 0, bool logits_required = true) {
    size_t n = start;
    for (const auto & step : schedule) {
        const size_t n0 = step.at("n0"), n1 = step.at("n1");
        require(n0 == n && n1 > n0 && n1 <= tokens.size() && n1 - n0 <= 256, "invalid replay tiling");
        llama_batch batch = llama_batch_init(n1 - n0, 0, 1);
        for (size_t i = n0; i < n1; ++i) {
            const size_t j = i - n0;
            batch.token[j] = tokens[i];
            batch.pos[j] = i;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = i == n1 - 1 && step.at("logits").get<bool>();
        }
        batch.n_tokens = n1 - n0;
        const int code = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(code == 0, "replay decode failed");
        n = n1;
    }
    require(n == tokens.size() && (!logits_required || schedule.back().at("logits").get<bool>()), "incomplete replay schedule");
}

static int check_checkpoint(const hybrid::json & fixture, llama_model * model, bool gpu, const char * output) {
    const auto tokens = fixture.at("tokens").get<std::vector<llama_token>>();
    const int k = fixture.at("k");
    const int64_t max_ctx = fixture.at("max_ctx");
    require(max_ctx > 0 && max_ctx <= 1048576 && uint64_t(max_ctx) >= tokens.size() + k + 2, "invalid snapshot context capacity");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int threads = fixture.value("threads", 2);
    require(threads >= 1 && threads <= 8, "invalid checkpoint threads");
    std::map<std::string, checkpoint_observation> observations;
    hybrid::json result = {{"stage", "checkpoint"}, {"depth", tokens.size()}, {"k", k},
        {"raw_logits", {{"dtype", "float32-le"}, {"vocab", vocab}, {"files", hybrid::json::object()}}}};
    auto record = [&](const std::string & name, checkpoint_observation observation) {
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "raw logits require little endian host");
        const auto path = std::string(output) + "." + name + ".f32";
        std::ofstream raw(path, std::ios::binary);
        raw.write(reinterpret_cast<const char *>(observation.logits.data()), observation.logits.size() * sizeof(float));
        raw.close();
        require(raw.good(), "cannot write raw checkpoint logits");
        result["raw_logits"]["files"][name] = {{"path", path}, {"sha256", hybrid::file_hash(path.c_str())}};
        observations[name] = std::move(observation);
    };
    if (fixture.contains("snapshots")) {
#ifdef LLAMA_HYBRID_SERVER_SNAPSHOTS
        const auto paths = fixture.at("snapshots").get<std::vector<std::string>>();
        const auto hashes = fixture.at("snapshot_sha256").get<std::vector<std::string>>();
        require((paths.size() == 2 || paths.size() == 3) && hashes.size() == paths.size(), "need hybrid, local and optional partition snapshots");
        // Score the live local first so every cell receives the same teacher-forced continuation.
        for (size_t i : {size_t(1), size_t(0), size_t(2)}) {
            if (i >= paths.size()) { continue; }
            auto ctx = context(model, max_ctx, gpu, threads);
            require(hybrid::file_hash(paths[i].c_str()) == hashes[i], "snapshot checksum mismatch");
            size_t count = 0;
            require(llama_state_seq_load_file(ctx.get(), paths[i].c_str(), 0, nullptr, 0, &count) > 0 && count <= tokens.size() + 4,
                    "invalid checkpoint token count");
            std::vector<llama_token> packed(count);
            require(llama_state_seq_load_file(ctx.get(), paths[i].c_str(), 0, packed.data(), packed.size(), &count) > 0,
                    "cannot load checkpoint snapshot");
            const auto loaded = server_tokens::deserialize(packed, false).get_text_tokens();
            require(loaded.size() == tokens.size() - 1 && std::equal(loaded.begin(), loaded.end(), tokens.begin()),
                    "snapshot token journal mismatch");
            hybrid::prefill(ctx.get(), {tokens.back()}, loaded.size());
            record(i == 0 ? "H" : i == 1 ? "L" : "P", observe(ctx.get(), model, tokens.size(), k,
                i == 1 ? std::vector<llama_token>() : observations.at("L").teacher));
        }
#else
        throw std::runtime_error("snapshot checker requires LLAMA_BUILD_SERVER=ON");
#endif
    }
    if (fixture.contains("replay")) {
        for (const std::string name : {"A", "B1", "B2", "B"}) {
            auto ctx = context(model, max_ctx, gpu, threads);
            replay_schedule(ctx.get(), tokens, fixture.at("replay").at(name));
            record(name, observe(ctx.get(), model, tokens.size(), k, observations.count("L") ? observations.at("L").teacher :
                name == "A" ? std::vector<llama_token>() : observations.at("A").teacher));
        }
        result["factorial"] = hybrid::json::object();
        for (const std::string name : {"A", "B1", "B2", "B"}) {
            result["factorial"]["A-" + name] = score_observations(observations.at("A"), observations.at(name), tokens.size(), k);
        }
        if (observations.count("H")) {
            result["factorial"]["B-H"] = score_observations(observations.at("B"), observations.at("H"), tokens.size(), k);
            result["factorial"]["A-L"] = score_observations(observations.at("A"), observations.at("L"), tokens.size(), k);
        }
    }
    if (fixture.contains("probes")) {
        require(fixture.at("probes").is_array() && fixture.at("probes").size() <= 16, "too many state probes");
        result["probes"] = hybrid::json::object();
        for (const auto & probe : fixture.at("probes")) {
            const std::string name = probe.at("name");
            require(!name.empty() && name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos,
                    "invalid state probe name");
            require(!observations.count(name), "duplicate state probe name");
            auto ctx = context(model, max_ctx, gpu, threads);
            size_t n = 0;
            bool ready = false;
            bool pending_delta = false;
            hybrid::delta_import pending;
            std::vector<llama_token> committed;
            hybrid::json states = hybrid::json::array();
            for (const auto & step : probe.at("steps")) {
                require(!pending_delta || step.contains("delta"), "missing final state probe chunk");
                if (step.contains("decode")) {
                    const auto & schedule = step.at("decode");
                    require(!schedule.empty(), "empty state probe decode");
                    const size_t end = schedule.back().at("n1");
                    require(end <= tokens.size(), "state probe decode exceeds journal");
                    replay_schedule(ctx.get(), {tokens.begin(), tokens.begin() + end}, schedule, n, false);
                    n = end;
                    ready = schedule.back().at("logits");
                } else if (step.contains("snapshot") && !step.contains("replace")) {
                    const std::string path = step.at("snapshot");
                    require(hybrid::file_hash(path.c_str()) == step.at("sha256"), "state probe snapshot checksum mismatch");
                    size_t count = 0;
                    std::vector<llama_token> journal(tokens.size());
                    require(llama_state_seq_load_file(ctx.get(), path.c_str(), 0, journal.data(), journal.size(), &count) > 0 &&
                            count <= tokens.size() && std::equal(journal.begin(), journal.begin() + count, tokens.begin()),
                            "state probe snapshot journal mismatch");
                    n = count;
                    ready = false;
                } else if (step.contains("delta")) {
                    const std::string path = step.at("delta");
                    require(hybrid::file_hash(path.c_str()) == step.at("sha256"), "state probe delta checksum mismatch");
                    auto wire = read_file(path.c_str());
                    const auto delta = hybrid::read_delta_view(wire, fixture.at("model_sha256"), n);
                    const size_t end = delta.header.at("n1");
                    require(end <= tokens.size() && delta.header.at("tokens").get<std::vector<llama_token>>() ==
                            std::vector<llama_token>(tokens.begin() + n, tokens.begin() + end), "state probe delta journal mismatch");
                    if (!pending_delta) { committed.assign(tokens.begin(), tokens.begin() + n); }
                    const bool final = pending.stage(ctx.get(), 0, std::move(wire), fixture.at("model_sha256"),
                        delta.header.at("session_id"), committed, max_ctx, llama_vocab_n_tokens(llama_model_get_vocab(model)));
                    if (final) {
                        pending.apply(ctx.get(), 0);
                        pending.clear();
                    }
                    pending_delta = !final;
                    n = end;
                    ready = false;
                } else if (step.contains("compare_snapshot")) {
                    const std::string path = step.at("compare_snapshot");
                    require(hybrid::file_hash(path.c_str()) == step.at("sha256"), "state probe snapshot checksum mismatch");
                    std::ifstream snapshot(path, std::ios::binary);
                    const auto state = hybrid::save_state(ctx.get());
                    states.push_back({{"n", n}, {"snapshot_bytes_equal", snapshot_state_matches(snapshot, n, state)}});
                } else if (step.contains("replace")) {
                    const std::string path = step.at("snapshot");
                    require(hybrid::file_hash(path.c_str()) == step.at("sha256"), "state probe snapshot checksum mismatch");
                    auto other = context(model, max_ctx, gpu, threads);
                    size_t count = 0;
                    std::vector<llama_token> journal(tokens.size());
                    require(llama_state_seq_load_file(other.get(), path.c_str(), 0, journal.data(), journal.size(), &count) > 0 &&
                            count == n && std::equal(journal.begin(), journal.begin() + count, tokens.begin()),
                            "state probe replacement journal mismatch");
                    auto state = hybrid::save_state(ctx.get());
                    const auto reference = hybrid::save_state(other.get());
                    const size_t split = 8 + attention_state_size(ctx.get());
                    require(split == 8 + attention_state_size(other.get()) && state.size() == reference.size() && split <= state.size(),
                            "state probe component layout mismatch");
                    const std::string component = step.at("replace");
                    require(component == "attention" || component == "recurrent", "invalid state probe component");
                    states.push_back({{"n", n}, {"replace", component},
                        {"attention_equal", state.substr(8, split - 8) == reference.substr(8, split - 8)},
                        {"recurrent_equal", state.substr(split) == reference.substr(split)}});
                    if (component == "attention") { state.replace(8, split - 8, reference, 8, split - 8); }
                    else { state.replace(split, state.size() - split, reference, split, reference.size() - split); }
                    hybrid::load_state(ctx.get(), state);
                    ready = false;
                } else if (step.contains("roundtrip")) {
                    const auto before = hybrid::save_state(ctx.get());
                    const std::string mode = step.at("roundtrip");
                    if (mode == "native") {
                        hybrid::load_state(ctx.get(), before);
                    } else {
                        require(mode == "range" && n > 0, "invalid state probe roundtrip");
                        const auto range = hybrid::export_range(ctx.get(), 0, n);
                        llama_memory_clear(llama_get_memory(ctx.get()), true);
                        require(llama_state_seq_set_data_append(ctx.get(), reinterpret_cast<const uint8_t *>(range.data()),
                                range.size(), 0, 0, n) == range.size(), "state probe roundtrip import failed");
                    }
                    states.push_back({{"n", n}, {"roundtrip", mode}, {"bytes_equal", before == hybrid::save_state(ctx.get())}});
                    ready = false;
                } else {
                    throw std::runtime_error("unknown state probe step");
                }
            }
            require(!pending_delta, "missing final state probe chunk");
            require(n == tokens.size() && ready, "incomplete state probe journal or final logits");
            record(name, observe(ctx.get(), model, n, k, observations.at("A").teacher));
            result["probes"][name]["roundtrips"] = states;
            for (const std::string reference : {"A", "B", "H"}) {
                if (observations.count(reference)) {
                    result["probes"][name][reference] = score_observations(observations.at(reference), observations.at(name), n, k);
                }
            }
        }
    }
    if (!observations.count("H")) {
        result["status"] = "replay_only";
        result["exit_code"] = 0;
    } else {
        result.update(score_observations(observations.at("L"), observations.at("H"), tokens.size(), k));
        bool accepted = result.at("exact");
        result["status_rule"] = "same_snapshot_exact";
        result["reference"] = "KL(local || hybrid), local greedy teacher forcing";
        if (observations.count("P")) {
            const auto partition = score_observations(observations.at("L"), observations.at("P"), tokens.size(), k);
            result["partition"] = partition;
            result["local_partition_mean_kl_nats"] = partition.at("mean_kl_nats");
            result["local_partition_p95_kl_nats"] = partition.at("p95_kl_nats");
            const double band = partition.at("mean_kl_nats");
            const double mean = result.at("mean_kl_nats");
            result["kl_limit"] = 0;
            result["kl_to_partition_ratio"] = band > 0 ? hybrid::json(mean / band) : hybrid::json(nullptr);
            result["kl_to_partition_ratio_status"] = band > 0 ? "finite" : mean == 0 ? "both_zero" : "zero_noise_band";
            result["status_rule"] = "session_mean_kl_zero";
            accepted = within_session_band(mean, 0, result.at("top1_agreed"), k, result.at("free_running_first_divergence"));
        }
        if (fixture.contains("acceptance_rule")) {
            require(fixture.at("acceptance_rule") == "session_append_replay_exact" && k == 64,
                    "invalid append replay acceptance rule or width");
            result["gate_controls"] = append_replay_controls(result);
            accepted = append_tokens_agree(result.at("top1_agreed"));
            for (const auto & control : result.at("gate_controls")) { accepted &= control.get<bool>(); }
            result["status_rule"] = "session_append_replay_exact";
            result["kl_limit"] = nullptr;
        }
        const bool heterogeneous = fixture.value("heterogeneous", false);
        result["status"] = heterogeneous ? "heterogeneous" : accepted ? "passed" : "failed";
        result["exit_code"] = heterogeneous ? 5 : accepted ? 0 : 3;
    }
    std::ofstream target(output);
    target << result.dump(2) << '\n';
    require(target.good(), "cannot write checkpoint result");
    std::cout << "checkpoint " << result.at("status") << '\n';
    return result.at("exit_code");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--test-backend-labels") {
            // Registry aliases normalize; real cross-backend fixtures still refuse.
            for (const auto & name : {"MTL", "Metal", "CUDA", "CPU"}) {
                const std::string expected = std::string(name) == "CUDA" ? "cuda" : std::string(name) == "CPU" ? "cpu" : "metal";
                require(checker_backend_label(name) == expected, ("backend_label_" + std::string(name)).c_str());
                std::cout << "PASS backend_label_" << name << '\n';
            }
            for (const auto & pair : {std::make_pair("MTL", "cuda"), std::make_pair("CUDA", "metal")}) {
                bool refused = false;
                const std::string expected = pair.second;
                try { checker_backend_label(pair.first, &expected); }
                catch (const std::runtime_error & error) { refused = std::string(error.what()) == "checker backend label/device mismatch"; }
                require(refused, ("fixture_" + expected + "_actual_" + pair.first + "_refuses").c_str());
                std::cout << "PASS fixture_" << pair.second << "_actual_" << pair.first << "_refuses\n";
            }
            return 0;
        }
        require(argc == 6, "usage: llama-hybrid-bench-check MODEL TOKENS.json DELTA OUTPUT.json GPU_LAYERS");
        // Full-vocabulary KL must detect a changed distribution, including one with unchanged top-1.
        const float p[] = {2, -2}, q[] = {-2, 2}, near[] = {1, 0};
        require(compare(p, p, 2) < 1e-12 && compare(p, q, 2) > .01 && compare(p, near, 2) > .01, "KL control failed");
        // The measured rule accepts its observed bound, rejects either excess, and keeps exact references bit-exact.
        require(within_repeat_floor(true, true, 0, 0, 0, 0) && !within_repeat_floor(false, true, 0, 0, 0, 0) &&
                within_repeat_floor(false, false, .01, .1, .01, .1) && !within_repeat_floor(false, false, .02, .1, .01, .1) &&
                !within_repeat_floor(false, false, .01, .2, .01, .1), "measured floor control failed");
        // Mean KL uses only a mean reference; top-1 loss and free-running divergence still fail.
        require(within_session_band(.001, .001, 63, 64, -1) && within_session_band(0, 0, 64, 64, -1) &&
                !within_session_band(.003, .001, 64, 64, -1) && !within_session_band(.0001, 0, 64, 64, -1) &&
                !within_session_band(.001, .001, 62, 64, -1) && !within_session_band(.001, .001, 64, 64, 63),
                "session mean acceptance control failed");
        // Snapshot byte checks preserve equality and reject changed or truncated payloads.
        std::istringstream snapshot_equal("0123456789012345xyz"), snapshot_changed("0123456789012345xyw"), snapshot_short("0123456789012345xy");
        const std::string native_state = "01234567xyz";
        require(snapshot_state_matches(snapshot_equal, 1, native_state) && !snapshot_state_matches(snapshot_changed, 1, native_state) &&
                !snapshot_state_matches(snapshot_short, 1, native_state), "snapshot byte comparison control failed");
        // Only teacher-forced top-1 gates L-H; free divergence and KL are reported.
        require(append_tokens_agree(63) && append_tokens_agree(64) && !append_tokens_agree(62),
                "append token acceptance control failed");
        const auto fixture = hybrid::json::parse(read_file(argv[2], 64 * 1024 * 1024), [](int depth, hybrid::json::parse_event_t, hybrid::json &) {
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
            backend = name;
            for (int i = 0; i < llama_model_n_devices(model.get()); ++i) {
                require(name == ggml_backend_reg_name(ggml_backend_dev_backend_reg(llama_model_get_device(model.get(), i))), "mixed checker devices unsupported");
            }
        }
        const std::string expected_backend = fixture.value("backend", checker_backend_label(backend));
        backend = checker_backend_label(backend, &expected_backend);
        const bool gpu = backend != "cpu";
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) { require(token >= 0 && token < vocab, "invalid check token"); }
        if (fixture.contains("snapshots") || fixture.contains("replay")) {
            return check_checkpoint(fixture, model.get(), gpu, argv[4]);
        }
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
        hybrid::delta_import pending;
        std::vector<llama_token> committed;
        for (const auto & path : paths) {
            const auto wire = read_file(path.c_str());
            const auto delta = hybrid::read_delta(wire, model_hash, n);
            const int64_t n1 = delta.header.at("n1");
            require(n1 <= int64_t(tokens.size() - 1), "check delta endpoint mismatch");
            require(delta.header.at("tokens").get<std::vector<llama_token>>() ==
                    std::vector<llama_token>(tokens.begin() + n, tokens.begin() + n1), "check delta tokens mismatch");
            require(delta.header.at("prefix_hash_n1") == hybrid::prefix_hash({tokens.begin(), tokens.begin() + n1}), "check prefix mismatch");
            const bool final = pending.stage(imported.get(), 0, wire, model_hash, delta.header.at("session_id").get<std::string>(), committed, ctx, vocab);
            if (final) {
                pending.apply(imported.get(), 0);
                committed = pending.imported_tokens();
                pending.clear();
            }
            hybrid::prefill(split.get(), {tokens.begin() + n, tokens.begin() + n1}, n);
            const auto repeat_start = std::chrono::steady_clock::now();
            hybrid::prefill(repeat.get(), {tokens.begin() + n, tokens.begin() + n1}, n);
            repeat_seconds += seconds_since(repeat_start);
            const auto ends = boundaries(n1, 256, n);
            split_ends.insert(split_ends.end(), ends.begin(), ends.end());
            n = n1;
        }
        require(committed.size() == size_t(n), "missing final check chunk");
        require(n == fixture.value("local_tail_start", int64_t(tokens.size() - 1)) && n < int64_t(tokens.size()), "incomplete check chain");
        const auto tail_ends = boundaries(tokens.size(), 256, n);
        split_ends.insert(split_ends.end(), tail_ends.begin(), tail_ends.end());
        const std::vector<llama_token> tail(tokens.begin() + n, tokens.end());
        hybrid::prefill(imported.get(), tail, n);
        hybrid::prefill(split.get(), tail, n);
        const auto repeat_start = std::chrono::steady_clock::now();
        hybrid::prefill(repeat.get(), tail, n);
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
