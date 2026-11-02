#include "hybrid-state.h"
#include "llama-model.h"

#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

#if defined(__GLIBC__)
// A no_alloc test stream supplies header bytes only and refuses every weight read.
struct header_stream {
    std::vector<char> bytes;
    int64_t position = 0;
    int64_t size = 0;
    bool data_read = false;

    static ssize_t read(void * cookie, char * dst, size_t count) {
        auto & stream = *static_cast<header_stream *>(cookie);
        if (stream.position >= (int64_t) stream.bytes.size()) {
            stream.data_read = true;
            errno = EIO;
            return -1;
        }
        count = std::min(count, stream.bytes.size() - (size_t) stream.position);
        memcpy(dst, stream.bytes.data() + stream.position, count);
        stream.position += count;
        return count;
    }

    static int seek(void * cookie, off64_t * offset, int whence) {
        auto & stream = *static_cast<header_stream *>(cookie);
        int64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? stream.position : stream.size;
        if (*offset < -base || *offset > stream.size - base) { errno = EINVAL; return -1; }
        stream.position = base + *offset;
        *offset = stream.position;
        return 0;
    }
};
#endif

int main(int argc, char ** argv) {
    try {
        const bool header_only = argc == 3 && std::string(argv[1]) == "--header-only";
        if (argc != 2 && !header_only) {
            throw std::runtime_error("usage: llama-hybrid-check-identity [--header-only] MODEL.gguf");
        }
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.no_alloc = true;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.load_mtp = false;
        hybrid::model_ptr model;
        if (header_only) {
#if defined(__GLIBC__)
            std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
            if (!input || input.tellg() > 32 * 1024 * 1024) { throw std::runtime_error("invalid header fixture size"); }
            header_stream stream;
            stream.bytes.resize((size_t) input.tellg());
            input.seekg(0);
            input.read(stream.bytes.data(), stream.bytes.size());
            stream.bytes.resize((stream.bytes.size() + 31) / 32 * 32, 0);
            gguf_init_params gp = {true, nullptr};
            std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
                gguf_init_from_buffer(stream.bytes.data(), stream.bytes.size(), gp), gguf_free);
            if (!header) { throw std::runtime_error("invalid header fixture"); }
            const int64_t count = gguf_get_n_tensors(header.get());
            stream.size = gguf_get_data_offset(header.get());
            if (count) {
                stream.size += gguf_get_tensor_offset(header.get(), count - 1) + gguf_get_tensor_size(header.get(), count - 1);
            }
            if (stream.size < 0 || stream.size > (1LL << 37)) { throw std::runtime_error("invalid fixture geometry"); }
            cookie_io_functions_t io = {header_stream::read, nullptr, header_stream::seek, nullptr};
            std::unique_ptr<FILE, decltype(&fclose)> file(fopencookie(&stream, "r", io), fclose);
            if (!file) { throw std::runtime_error("cannot open header stream"); }
            setvbuf(file.get(), nullptr, _IONBF, 0);
            model.reset(llama_model_load_from_file_ptr(file.get(), params));
            if (stream.data_read) { throw std::runtime_error("no_alloc attempted to read weights"); }
#else
            throw std::runtime_error("header-only fixture streams require glibc");
#endif
        } else {
            model.reset(llama_model_load_from_file(argv[1], params));
        }
        if (!model) { throw std::runtime_error("cannot load model metadata"); }
        const auto layout = hybrid::model_layout(model.get());
        // Skipped MTP tensors must not enter a graph or the transferred memory.
        for (uint32_t i = model->hparams.n_layer(); i < model->hparams.n_layer_all; ++i) {
            const auto & layer = model->layers[i];
            if (layer.wq || layer.wk || layer.wv || layer.wo || layer.ffn_up || layer.nextn.eh_proj) {
                throw std::runtime_error("MTP weights were loaded");
            }
        }
        std::cout << hybrid::json({{"model_layout", layout},
                                  {"layout_fingerprint", hybrid::layout_fingerprint(layout)}}).dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
