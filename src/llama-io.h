#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Opt-in trace records are buffered; only the writer thread performs trace I/O.
struct llama_trace_context {
    char call_id[129] = {};
    char session_id[129] = {};
    char transfer_id[129] = {};
    char parent_id[129] = {};
    char parent_domain[129] = {};
    char request_id[129] = {};
    char request_domain[129] = {};
    char origin[129] = {};
    int64_t chunk_id = -1, n0 = -1, n1 = -1;
};

LLAMA_API void llama_trace_start();
LLAMA_API bool llama_trace_enabled();
LLAMA_API bool llama_trace_checkpoint();
LLAMA_API uint64_t llama_trace_now();
LLAMA_API llama_trace_context llama_trace_get();
LLAMA_API void llama_trace_set(const llama_trace_context & context);
LLAMA_API void llama_trace_header(llama_trace_context & context, const std::string & key, const std::string & value);

struct llama_trace_binding {
    llama_trace_context previous;
    explicit llama_trace_binding(const llama_trace_context & context) : previous(llama_trace_get()) { llama_trace_set(context); }
    ~llama_trace_binding() { llama_trace_set(previous); }
};

struct LLAMA_API llama_trace_span {
    const char * name;
    const char * backend = "unspecified";
    uint64_t id = 0, start_ns = 0;
    size_t bytes = 0, tokens = 0;
    llama_trace_context previous;
    explicit llama_trace_span(const char * name, size_t bytes = 0, size_t tokens = 0, uint64_t start = 0);
    ~llama_trace_span();
    llama_trace_span(const llama_trace_span &) = delete;
    llama_trace_span & operator=(const llama_trace_span &) = delete;
};

// Opt-in import timings include nested stages; tensor I/O includes backend waits.
struct llama_io_timer {
    llama_trace_span trace;
    const char * stage;
    size_t bytes;
    const bool enabled = !llama_trace_enabled() && std::getenv("LLAMA_HYBRID_IMPORT_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point start;
    llama_io_timer(const char * stage, size_t bytes = 0) : trace(stage, bytes), stage(stage), bytes(bytes) {
        if (enabled) { start = std::chrono::steady_clock::now(); }
    }
    ~llama_io_timer() {
        if (enabled) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::fprintf(stderr, "hybrid_import stage=%s bytes=%zu ms=%.6f\n", stage, bytes, ms);
        }
    }
};

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
