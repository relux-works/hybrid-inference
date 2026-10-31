#include "hybrid-state.h"
#include "hybrid-scheduler.h"
#include "httplib.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>

using hybrid::json;

struct session {
    std::vector<llama_token> tokens;
    std::string state;
    int64_t last_used = 0;
    uint64_t recency = 0;
};

class response_budget {
    std::mutex mutex;
    size_t active = 0;
    std::map<std::string, size_t> clients;

    struct lease {
        response_budget & owner;
        std::string client;
        lease(response_budget & owner, const std::string & client) : owner(owner), client(client) {}
        ~lease() {
            std::lock_guard<std::mutex> lock(owner.mutex);
            --owner.active;
            --owner.clients.at(client);
        }
    };

public:
    std::shared_ptr<void> acquire(const std::string & client) {
        std::lock_guard<std::mutex> lock(mutex);
        // Reserve one of the four HTTP workers for admission and control requests.
        if (active >= 3 || clients[client] >= 2) { return {}; }
        auto permit = std::make_shared<lease>(*this, client);
        ++active;
        ++clients[client];
        return permit;
    }
};

static void send_binary(httplib::Response & res, std::string wire) {
    res.set_chunked_content_provider("application/octet-stream", [wire = std::move(wire)](size_t offset, httplib::DataSink & sink) {
        size_t n = std::min<size_t>(65536, wire.size() - offset);
        if (n && !sink.write(wire.data() + offset, n)) { return false; }
        if (offset + n == wire.size()) { sink.done(); }
        return true;
    });
}

static int64_t now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static void respond(httplib::Response & res, int status, const json & value) {
    res.status = status;
    res.set_content(value.dump(), "application/json");
}

static std::string new_id() {
    std::random_device rng;
    static const char digits[] = "0123456789abcdef";
    std::string id;
    for (int i = 0; i < 32; ++i) { id += digits[rng() & 15]; }
    return id;
}

static json info(const session & s, const std::string & model_hash) {
    return {{"n", s.tokens.size()}, {"prefix_hash", hybrid::prefix_hash(s.tokens)}, {"tier", "ram"}, {"last_used", s.last_used}, {"model_sha256", model_hash}};
}

static std::string snapshot(const session & s, const std::string & empty_state) {
    std::string out;
    auto u32 = [&](uint32_t n) {
        for (int i = 0; i < 4; ++i) { out += (char) (n >> (8 * i)); }
    };
    u32(LLAMA_STATE_SEQ_MAGIC);
    u32(LLAMA_STATE_SEQ_VERSION);
    u32(s.tokens.size());
    for (auto token : s.tokens) { u32(token); }
    const auto & state = s.state.empty() ? empty_state : s.state;
    if (state.size() < 8) { throw std::runtime_error("invalid snapshot state"); }
    out.append(state.data() + 8, state.size() - 8);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: llama-hybrid-prefill MODEL.gguf CLIENTS.json [PORT=8091] [N_CTX=8448] [--host ADDRESS] [--gpu-layers N] [--bench] [--max-ctx N=16384] [--ram-budget-mib N=512] [--max-request-mib N=1]\n";
        return 2;
    }
    try {
        int port = 8091;
        int n_ctx = 8448;
        int max_ctx = 16384;
        size_t ram_budget = 512 * 1024 * 1024;
        size_t max_request = 1024 * 1024;
        std::string host = "127.0.0.1";
        int gpu_layers = 0;
        bool bench = false;
        bool context_given = false;
        bool max_ctx_given = false;
        auto number = [](const std::string & text, uint64_t maximum) {
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) { throw std::runtime_error("invalid number"); }
            const auto n = std::stoull(text);
            if (n > maximum) { throw std::runtime_error("number out of range"); }
            return n;
        };
        int first_option = 3;
        if (first_option < argc && argv[first_option][0] != '-') { port = number(argv[first_option++], 65535); }
        if (first_option < argc && argv[first_option][0] != '-') {
            n_ctx = number(argv[first_option++], INT32_MAX - 256);
            context_given = true;
        }
        for (int i = first_option; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--bench") { bench = true; }
            else if (arg == "--host" && i + 1 < argc) { host = argv[++i]; }
            else if (arg == "--gpu-layers" && i + 1 < argc) { gpu_layers = number(argv[++i], 999); }
            else if (arg == "--max-ctx" && i + 1 < argc) {
                max_ctx = number(argv[++i], INT32_MAX - 256);
                max_ctx_given = true;
            } else if ((arg == "--ram-budget-mib" || arg == "--max-request-mib") && i + 1 < argc) {
                const size_t value = number(argv[++i], std::numeric_limits<size_t>::max() / (1024 * 1024));
                if (value == 0) { throw std::runtime_error("budget must be positive"); }
                if (arg == "--ram-budget-mib") { ram_budget = value * 1024 * 1024; }
                else { max_request = value * 1024 * 1024; }
            } else { throw std::runtime_error("unknown option"); }
        }
        if (max_ctx_given && !context_given) { n_ctx = max_ctx; }
        if (port < 1 || n_ctx < 256 || n_ctx > max_ctx) { throw std::runtime_error("invalid port or context size"); }
        std::ifstream credentials(argv[2]);
        std::string text((std::istreambuf_iterator<char>(credentials)), {});
        json clients = hybrid::parse_json(text);
        if (!clients.is_object() || clients.empty()) { throw std::runtime_error("clients must be a nonempty id-to-token object"); }
        std::map<std::string, std::string> owners;
        for (auto it = clients.begin(); it != clients.end(); ++it) {
            if (!it.value().is_string()) { throw std::runtime_error("client token must be a string"); }
            std::string token = it.value();
            if (token.size() < 16 || token.size() > 256 || !owners.emplace("Bearer " + token, it.key()).second) {
                throw std::runtime_error("client tokens must be distinct and 16-256 bytes long");
            }
        }
        const std::string model_hash = hybrid::file_hash(argv[1]);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        if (!model || !llama_model_is_hybrid(model.get())) { throw std::runtime_error("a supported hybrid model is required"); }
        const json layout = hybrid::model_layout(model.get());
        const std::string fingerprint = hybrid::layout_fingerprint(layout);
        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = cp.n_ubatch = 256;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = gpu_layers > 0;
        hybrid::context_ptr ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) { throw std::runtime_error("cannot create context"); }
        const std::string empty_state = hybrid::save_state(ctx.get());
        std::map<std::string, std::map<std::string, std::shared_ptr<session>>> sessions;
        uint64_t recency = 0;
        hybrid::request_scheduler scheduler;
        std::atomic<size_t> ram_bytes{0};
        std::atomic<uint64_t> extends{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<size_t> session_count{0};
        response_budget responses;
        httplib::Server http;
        http.set_payload_max_length(max_request);
        http.set_read_timeout(30, 0);
        http.set_write_timeout(3600, 0);
        http.set_keep_alive_max_count(1);
        http.new_task_queue = [] { return new httplib::ThreadPool(4); };

        auto handler = [&](const httplib::Request & req, httplib::Response & res) {
            if (req.path == "/v1/health") {
                respond(res, 200, {{"status", "ok"}, {"model_sha256", model_hash}, {"layout_fingerprint", fingerprint},
                                   {"model_layout", layout}, {"max_ctx", max_ctx}, {"n_ctx", n_ctx},
                                   {"ram_budget_bytes", ram_budget}, {"ram_bytes", ram_bytes.load()},
                                   {"sessions", session_count.load()}, {"evictions", evictions.load()},
                                   {"max_request_bytes", max_request}});
                return;
            }
            if (req.path == "/metrics") {
                res.set_content("hybrid_extends_total " + std::to_string(extends.load()) +
                                "\nhybrid_ram_bytes " + std::to_string(ram_bytes.load()) +
                                "\nhybrid_max_ctx " + std::to_string(max_ctx) +
                                "\nhybrid_ram_budget_bytes " + std::to_string(ram_budget) +
                                "\nhybrid_sessions " + std::to_string(session_count.load()) +
                                "\nhybrid_evictions " + std::to_string(evictions.load()) +
                                "\nhybrid_max_request_bytes " + std::to_string(max_request) +
                                "\nhybrid_model_info{model_sha256=\"" + model_hash + "\",layout_fingerprint=\"" + fingerprint + "\"} 1\n", "text/plain");
                return;
            }
            auto owner = owners.find(req.get_header_value("Authorization"));
            if (owner == owners.end()) {
                res.set_header("WWW-Authenticate", "Bearer");
                respond(res, 401, {{"error", "unauthorized"}});
                return;
            }
            if (bench && req.path == "/v1/bench/link") {
                res.set_content(std::string(1024 * 1024, 'x'), "application/octet-stream");
                return;
            }
            const std::string & client = owner->second;
            auto permit = responses.acquire(client);
            if (!permit) {
                res.set_header("Retry-After", "1");
                respond(res, 429, {{"error", "binary response limit"}});
                return;
            }
            // Response ownership covers queued work and every response type, including errors.
            res.user_data.set("hybrid_permit", std::move(permit));
            const std::string session_id = req.path.size() >= 45 ? req.path.substr(13, 32) : "";
            bool critical = true;
            if (req.method == "POST" && req.path.find("/extend") != std::string::npos) {
                try { critical = hybrid::parse_json(req.body, max_request).value("return_delta", true); }
                catch (const std::exception &) { /* Invalid requests are rejected below. */ }
            }
            auto admission = scheduler.enter(client, session_id, critical, req.is_connection_closed);
            if (!admission) { return; }
            try {
                auto & space = sessions[owner->second];
                const std::string base = "/v1/sessions";
                if (req.path == base) {
                    if (req.method == "GET") {
                        json list = json::array();
                        for (const auto & entry : space) {
                            const auto & s = *entry.second;
                            list.push_back({{"session_id", entry.first}, {"n", s.tokens.size()}, {"tier", "ram"}, {"last_used", s.last_used}});
                        }
                        respond(res, 200, list);
                    } else {
                        auto body = hybrid::parse_json(req.body, max_request);
                        if (body.at("model_sha256") != model_hash) {
                            respond(res, 400, {{"error", "model hash mismatch"}});
                            return;
                        }
                        size_t count = 0;
                        for (const auto & client : sessions) { count += client.second.size(); }
                        if (count >= 128) { respond(res, 503, {{"error", "session limit"}}); return; }
                        std::string id;
                        do { id = new_id(); } while (space.count(id));
                        auto created = std::make_shared<session>();
                        created->last_used = now();
                        created->recency = ++recency;
                        space[id] = std::move(created);
                        ++session_count;
                        respond(res, 200, {{"session_id", id}});
                    }
                    return;
                }
                const std::string rest = req.path.substr(base.size() + 1);
                const auto slash = rest.find('/');
                const std::string id = rest.substr(0, slash);
                const std::string action = slash == std::string::npos ? "" : rest.substr(slash + 1);
                auto found = space.find(id);
                if (found == space.end()) { respond(res, 404, {{"error", "session not found"}}); return; }
                // Pin until the response is destroyed, including a slow binary reader.
                res.user_data.set("hybrid_session", found->second);
                session & s = *found->second;
                if (req.method == "DELETE" && action.empty()) {
                    ram_bytes -= s.state.size() + s.tokens.size() * sizeof(llama_token);
                    space.erase(found);
                    --session_count;
                    res.status = 204;
                } else if (req.method == "GET" && action.empty()) {
                    s.recency = ++recency;
                    respond(res, 200, info(s, model_hash));
                } else if (req.method == "GET" && action == "snapshot") {
                    s.recency = ++recency;
                    send_binary(res, snapshot(s, empty_state));
                } else if (req.method == "POST" && action == "extend") {
                    json body = hybrid::parse_json(req.body, max_request);
                    if (!body.at("n0").is_number_integer() || !body.at("prefix_hash_n0").is_string()) {
                        respond(res, 400, {{"error", "invalid append boundary"}}); return;
                    }
                    if (body.at("n0").get<int64_t>() != (int64_t) s.tokens.size() || body.at("prefix_hash_n0") != hybrid::prefix_hash(s.tokens)) {
                        respond(res, 409, {{"n", s.tokens.size()}, {"prefix_hash", hybrid::prefix_hash(s.tokens)}});
                        return;
                    }
                    const auto & input = body.at("tokens");
                    if (!input.is_array() || input.empty() || input.size() > (size_t) n_ctx - s.tokens.size() || !body.at("return_delta").is_boolean()) {
                        respond(res, 400, {{"error", "invalid tokens, context limit or return_delta"}}); return;
                    }
                    std::vector<llama_token> tokens;
                    int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
                    for (const auto & token : input) {
                        if (!token.is_number_integer() || token.get<int64_t>() < 0 || token.get<int64_t>() >= vocab) {
                            respond(res, 400, {{"error", "token out of vocabulary"}}); return;
                        }
                        tokens.push_back(token.get<llama_token>());
                    }
                    const size_t n0 = s.tokens.size();
                    std::vector<llama_token> all = s.tokens;
                    all.insert(all.end(), tokens.begin(), tokens.end());
                    const bool return_delta = body.at("return_delta").get<bool>();
                    json header;
                    if (return_delta) {
                        header = {{"model_sha256", model_hash}, {"session_id", id}, {"n0", n0}, {"n1", all.size()}, {"tokens", tokens}, {"prefix_hash_n1", hybrid::prefix_hash(all)}};
                        // Use the largest body length so the header budget is checked before inference.
                        hybrid::encode_delta_header(header, hybrid::max_delta_body_bytes);
                    }
                    hybrid::load_state(ctx.get(), s.state);
                    const auto prefill_started = std::chrono::steady_clock::now();
                    hybrid::prefill(ctx.get(), tokens, n0);
                    if (bench && req.get_header_value("X-Hybrid-Bench") == "1") {
                        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_started).count();
                        res.set_header("X-Hybrid-Prefill-Ms", std::to_string(ms));
                    }
                    std::string wire;
                    if (return_delta) {
                        wire = hybrid::encode_delta(std::move(header), hybrid::export_range(ctx.get(), n0, all.size()));
                    }
                    std::string state = hybrid::save_state(ctx.get());
                    size_t old_bytes = s.state.size() + s.tokens.size() * sizeof(llama_token);
                    size_t new_bytes = state.size() + all.size() * sizeof(llama_token);
                    // Plan before deleting: an impossible admission must preserve every session.
                    const size_t retained = ram_bytes.load() - old_bytes;
                    struct victim { std::string client; std::string id; uint64_t recency; size_t bytes; };
                    std::vector<victim> candidates;
                    for (const auto & client_space : sessions) {
                        for (const auto & entry : client_space.second) {
                            if (entry.second.use_count() != 1) { continue; }
                            const auto & idle = *entry.second;
                            candidates.push_back({client_space.first, entry.first, idle.recency,
                                                  idle.state.size() + idle.tokens.size() * sizeof(llama_token)});
                        }
                    }
                    std::sort(candidates.begin(), candidates.end(), [](const victim & a, const victim & b) { return a.recency < b.recency; });
                    size_t freed = 0;
                    size_t victims = 0;
                    if (new_bytes <= ram_budget) {
                        while (retained - freed > ram_budget - new_bytes && victims < candidates.size()) {
                            freed += candidates[victims++].bytes;
                        }
                    }
                    if (new_bytes > ram_budget || retained - freed > ram_budget - new_bytes) {
                        size_t reclaimable = 0;
                        for (const auto & candidate : candidates) { reclaimable += candidate.bytes; }
                        respond(res, 503, {{"error", "ram budget"}, {"need", retained - reclaimable + new_bytes}, {"budget", ram_budget}});
                        return;
                    }
                    for (size_t i = 0; i < victims; ++i) {
                        sessions.at(candidates[i].client).erase(candidates[i].id);
                    }
                    evictions += victims;
                    session_count -= victims;
                    ram_bytes = retained - freed + new_bytes;
                    s.tokens = std::move(all);
                    s.state = std::move(state);
                    s.last_used = now();
                    s.recency = ++recency;
                    ++extends;
                    if (!wire.empty()) {
                        send_binary(res, std::move(wire));
                    } else {
                        respond(res, 202, {{"n", s.tokens.size()}, {"prefix_hash", hybrid::prefix_hash(s.tokens)}});
                    }
                } else {
                    respond(res, 404, {{"error", "unknown endpoint"}});
                }
            } catch (const json::exception &) {
                respond(res, 400, {{"error", "invalid JSON fields"}});
            } catch (const std::exception &) {
                respond(res, 400, {{"error", "request could not be applied"}});
            }
        };
        http.Get("/v1/health", handler);
        http.Get("/metrics", handler);
        if (bench) { http.Get("/v1/bench/link", handler); }
        http.Get("/v1/sessions", handler);
        http.Post("/v1/sessions", handler);
        http.Get(R"(/v1/sessions/[a-f0-9]{32}(/snapshot)?)", handler);
        http.Post(R"(/v1/sessions/[a-f0-9]{32}/extend)", handler);
        http.Delete(R"(/v1/sessions/[a-f0-9]{32})", handler);
        std::cerr << "hybrid prefill listening on " << host << ":" << port << "\n";
        return http.listen(host, port) ? 0 : 1;
    } catch (const std::exception &) {
        std::cerr << "prefill startup failed; check model, client file and arguments\n";
        return 1;
    }
}
