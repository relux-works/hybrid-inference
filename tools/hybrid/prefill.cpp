#include "hybrid-state.h"
#include "httplib.h"
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
};

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
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: llama-hybrid-prefill MODEL.gguf CLIENTS.json [PORT=8091] [N_CTX=8448]\n";
        return 2;
    }
    try {
        const int port = argc > 3 ? std::stoi(argv[3]) : 8091;
        const int n_ctx = argc > 4 ? std::stoi(argv[4]) : 8448;
        if (port < 1 || port > 65535 || n_ctx < 256 || n_ctx > 16384) { throw std::runtime_error("invalid port or context size"); }
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
        mp.n_gpu_layers = 0;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], mp));
        if (!model || !llama_model_is_hybrid(model.get())) { throw std::runtime_error("a supported hybrid model is required"); }
        auto ctx = hybrid::make_context(model.get(), n_ctx);
        const std::string empty_state = hybrid::save_state(ctx.get());
        std::map<std::string, std::map<std::string, session>> sessions;
        std::mutex mutex;
        size_t ram_bytes = 0;
        uint64_t extends = 0;
        httplib::Server http;
        http.set_payload_max_length(1024 * 1024);
        http.set_read_timeout(30, 0);
        http.set_write_timeout(3600, 0);
        http.new_task_queue = [] { return new httplib::ThreadPool(4); };

        auto handler = [&](const httplib::Request & req, httplib::Response & res) {
            std::lock_guard<std::mutex> lock(mutex);
            try {
                if (req.path == "/v1/health") {
                    respond(res, 200, {{"status", "ok"}, {"model_sha256", model_hash}});
                    return;
                }
                if (req.path == "/metrics") {
                    res.set_content("hybrid_extends_total " + std::to_string(extends) + "\nhybrid_ram_bytes " + std::to_string(ram_bytes) + "\n", "text/plain");
                    return;
                }
                auto owner = owners.find(req.get_header_value("Authorization"));
                if (owner == owners.end()) {
                    res.set_header("WWW-Authenticate", "Bearer");
                    respond(res, 401, {{"error", "unauthorized"}});
                    return;
                }
                auto & space = sessions[owner->second];
                const std::string base = "/v1/sessions";
                if (req.path == base) {
                    if (req.method == "GET") {
                        json list = json::array();
                        for (const auto & entry : space) {
                            const auto & s = entry.second;
                            list.push_back({{"session_id", entry.first}, {"n", s.tokens.size()}, {"tier", "ram"}, {"last_used", s.last_used}});
                        }
                        respond(res, 200, list);
                    } else {
                        auto body = hybrid::parse_json(req.body);
                        if (body.at("model_sha256") != model_hash) {
                            respond(res, 400, {{"error", "model hash mismatch"}});
                            return;
                        }
                        size_t count = 0;
                        for (const auto & client : sessions) { count += client.second.size(); }
                        if (count >= 128) { respond(res, 503, {{"error", "session limit"}}); return; }
                        std::string id;
                        do { id = new_id(); } while (space.count(id));
                        space[id].last_used = now();
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
                session & s = found->second;
                if (req.method == "DELETE" && action.empty()) {
                    ram_bytes -= s.state.size() + s.tokens.size() * sizeof(llama_token);
                    space.erase(found);
                    res.status = 204;
                } else if (req.method == "GET" && action.empty()) {
                    respond(res, 200, info(s, model_hash));
                } else if (req.method == "GET" && action == "snapshot") {
                    res.set_content(snapshot(s, empty_state), "application/octet-stream");
                } else if (req.method == "POST" && action == "extend") {
                    json body = hybrid::parse_json(req.body);
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
                    hybrid::load_state(ctx.get(), s.state);
                    hybrid::prefill(ctx.get(), tokens, n0);
                    std::vector<llama_token> all = s.tokens;
                    all.insert(all.end(), tokens.begin(), tokens.end());
                    std::string wire;
                    if (body.at("return_delta").get<bool>()) {
                        wire = hybrid::encode_delta({{"model_sha256", model_hash}, {"session_id", id}, {"n0", n0}, {"n1", all.size()}, {"tokens", tokens}, {"prefix_hash_n1", hybrid::prefix_hash(all)}}, hybrid::export_range(ctx.get(), n0, all.size()));
                    }
                    std::string state = hybrid::save_state(ctx.get());
                    size_t old_bytes = s.state.size() + s.tokens.size() * sizeof(llama_token);
                    size_t new_bytes = state.size() + all.size() * sizeof(llama_token);
                    if (ram_bytes - old_bytes + new_bytes > 512 * 1024 * 1024) {
                        respond(res, 503, {{"error", "RAM session budget exceeded"}}); return;
                    }
                    ram_bytes = ram_bytes - old_bytes + new_bytes;
                    s.tokens = std::move(all);
                    s.state = std::move(state);
                    s.last_used = now();
                    ++extends;
                    if (!wire.empty()) {
                        res.set_chunked_content_provider("application/octet-stream", [wire = std::move(wire)](size_t offset, httplib::DataSink & sink) {
                            size_t n = std::min<size_t>(65536, wire.size() - offset);
                            if (n && !sink.write(wire.data() + offset, n)) { return false; }
                            if (offset + n == wire.size()) { sink.done(); }
                            return true;
                        });
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
        http.Get("/v1/sessions", handler);
        http.Post("/v1/sessions", handler);
        http.Get(R"(/v1/sessions/[a-f0-9]{32}(/snapshot)?)", handler);
        http.Post(R"(/v1/sessions/[a-f0-9]{32}/extend)", handler);
        http.Delete(R"(/v1/sessions/[a-f0-9]{32})", handler);
        std::cerr << "hybrid prefill listening on 127.0.0.1:" << port << "\n";
        return http.listen("127.0.0.1", port) ? 0 : 1;
    } catch (const std::exception &) {
        std::cerr << "prefill startup failed; check model, client file and arguments\n";
        return 1;
    }
}
