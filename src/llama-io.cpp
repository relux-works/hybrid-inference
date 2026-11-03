#include "llama-io.h"

#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
std::string trace_domain() {
    char host[64] = "host";
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
    if (gethostname(host, sizeof(host)-1) != 0) { std::strcpy(host, "host"); }
#endif
    return std::string(host) + ':' + std::to_string(pid) + ":native.steady_clock";
}
struct trace_record {
    const char * name;
    char backend[129];
    uint64_t id, start, end;
    size_t bytes, tokens;
    llama_trace_context context;
};

class trace_writer {
    static constexpr size_t capacity = 262144;
    struct cell { std::atomic<bool> ready{false}; trace_record record; };
    std::unique_ptr<cell[]> buffer{new cell[capacity]};
    std::atomic<uint64_t> head{0}, tail{0}, lost{0};
    std::atomic<bool> stopping{false};
    std::atomic<uint64_t> checkpoint_request{0}, checkpoint_done{0};
    std::atomic<bool> checkpoint_ok{true};
    std::ofstream file;
    std::thread worker;
public:
    const std::string domain;
    explicit trace_writer(const char * path) : file(path, std::ios::out | std::ios::app),
        domain(trace_domain()) {
        if (!file.good()) { throw std::runtime_error("cannot open native trace"); }
        file << "{\"kind\":\"trace_start\",\"clock_domain\":\"" << domain << "\"}\n";
        worker = std::thread([this] {
            while (!stopping.load() || tail.load() != head.load()) {
                auto index = tail.load();
                if (index == head.load() && checkpoint_request.load() != checkpoint_done.load()) {
                    file << "{\"kind\":\"trace_checkpoint\",\"dropped\":" << lost.load()
                         << ",\"records\":" << index << ",\"writer_ok\":" << (file.good() ? "true" : "false") << "}\n";
                    file.flush();
                    checkpoint_ok.store(file.good() && lost.load() == 0);
                    checkpoint_done.store(checkpoint_request.load());
                }
                if (index == head.load() || !buffer[index % capacity].ready.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                auto & slot = buffer[index % capacity];
                const auto & r = slot.record;
                const auto & c = r.context;
                file << "{\"kind\":\"span\",\"clock_domain\":\"" << domain << "\",\"id\":\"" << r.id
                     << "\",\"parent_id\":\"" << c.parent_id << "\",\"parent_domain\":\"" << c.parent_domain
                     << "\",\"parent_relation\":\"" << (c.parent_domain == domain ? "nested" : "causal_remote")
                     << "\",\"name\":\"" << r.name << "\",\"backend\":\"" << r.backend
                     << "\",\"call_id\":\"" << c.call_id << "\",\"session_id\":\"" << c.session_id
                     << "\",\"transfer_id\":\"" << c.transfer_id << "\",\"request_id\":\"" << c.request_id
                     << "\",\"request_domain\":\"" << c.request_domain << "\",\"n0\":" << c.n0 << ",\"n1\":" << c.n1 << ",\"chunk_id\":" << c.chunk_id
                     << ",\"origin\":\"" << c.origin << "\",\"start_ns\":" << r.start << ",\"end_ns\":" << r.end
                     << ",\"duration_ns\":" << r.end-r.start << ",\"bytes\":" << r.bytes << ",\"tokens\":" << r.tokens << "}\n";
                slot.ready.store(false, std::memory_order_release);
                tail.store(index + 1, std::memory_order_release);
            }
            file << "{\"kind\":\"trace_end\",\"dropped\":" << lost.load() << ",\"writer_ok\":" << (file.good() ? "true" : "false") << "}\n";
            file.flush();
        });
    }
    ~trace_writer() { stopping.store(true); worker.join(); }
    bool checkpoint() {
        auto sequence = checkpoint_request.fetch_add(1) + 1;
        while (checkpoint_done.load() < sequence) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        return checkpoint_ok.load();
    }
    void push(const trace_record & record) {
        auto at = head.load(std::memory_order_relaxed);
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (at - tail.load(std::memory_order_acquire) >= capacity) { ++lost; return; }
            if (head.compare_exchange_weak(at, at + 1)) {
                auto & cell = buffer[at % capacity];
                cell.record = record;
                cell.ready.store(true, std::memory_order_release);
                return;
            }
        }
        ++lost;
    }
};

std::unique_ptr<trace_writer> writer;
std::atomic<uint64_t> trace_id{1};
thread_local llama_trace_context trace_context;

void trace_text(char * dst, const std::string & text) {
    if (text.size() > 128 || text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:._-") != std::string::npos) {
        dst[0] = 0;
        return;
    }
    std::memcpy(dst, text.c_str(), text.size() + 1);
}
}

void llama_trace_start() {
    const char * path = std::getenv("LLAMA_HYBRID_TRACE");
    if (path && *path && !writer) {
        std::string expanded = path;
        const auto marker = expanded.find("%p");
#ifdef _WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        if (marker != std::string::npos) { expanded.replace(marker, 2, std::to_string(pid)); }
        if (std::ifstream(expanded).good()) { throw std::runtime_error("native trace already exists"); }
        writer.reset(new trace_writer(expanded.c_str()));
    }
}

bool llama_trace_enabled() { return bool(writer); }
bool llama_trace_checkpoint() { return writer && writer->checkpoint(); }
uint64_t llama_trace_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
llama_trace_context llama_trace_get() { return trace_context; }
void llama_trace_set(const llama_trace_context & context) { trace_context = context; }

void llama_trace_header(llama_trace_context & c, const std::string & key, const std::string & value) {
    std::string name = key;
    for (auto & ch : name) { if (ch >= 'A' && ch <= 'Z') { ch += 'a' - 'A'; } }
    if (name == "x-hybrid-trace-call")     { trace_text(c.call_id, value); }
    if (name == "x-hybrid-trace-parent")   { trace_text(c.parent_id, value); trace_text(c.request_id, value); }
    if (name == "x-hybrid-trace-domain")   { trace_text(c.parent_domain, value); trace_text(c.request_domain, value); }
    if (name == "x-hybrid-trace-origin")   { trace_text(c.origin, value); }
    if (name == "x-hybrid-trace-session")  { trace_text(c.session_id, value); }
    if (name == "x-hybrid-trace-transfer") { trace_text(c.transfer_id, value); }
    if ((name == "x-hybrid-trace-n0" || name == "x-hybrid-trace-n1") && !value.empty() && value.size() < 10 && value.find_first_not_of("0123456789") == std::string::npos) {
        if (name.back() == '0') { c.n0 = std::stoll(value); } else { c.n1 = std::stoll(value); }
    }
    if (name == "x-hybrid-trace-chunk" && !value.empty() && value.size() < 10 && value.find_first_not_of("0123456789") == std::string::npos) {
        c.chunk_id = std::stoll(value);
    }
}

llama_trace_span::llama_trace_span(const char * name, size_t bytes, size_t tokens, uint64_t start) : name(name), bytes(bytes), tokens(tokens) {
    if (writer && trace_context.call_id[0]) {
        previous = trace_context;
        id = trace_id.fetch_add(1);
        start_ns = start ? start : llama_trace_now();
        trace_text(trace_context.parent_id, std::to_string(id));
        trace_text(trace_context.parent_domain, writer->domain);
    }
}

llama_trace_span::~llama_trace_span() {
    if (id) {
        trace_record record;
        record.name = name;
        trace_text(record.backend, backend);
        record.id = id;
        record.start = start_ns;
        record.end = llama_trace_now();
        record.bytes = bytes;
        record.tokens = tokens;
        record.context = previous;
        writer->push(record);
        trace_context = previous;
    }
}
