#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Opt-in import timings include nested stages; tensor I/O includes backend waits.
struct llama_io_timer {
    const char * stage;
    size_t bytes;
    const bool enabled = std::getenv("LLAMA_HYBRID_IMPORT_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point start;
    llama_io_timer(const char * stage, size_t bytes = 0) : stage(stage), bytes(bytes) {
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
