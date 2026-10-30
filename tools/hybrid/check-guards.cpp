#include "hybrid-state.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

class state_reader : public llama_io_read_i {
public:
    state_reader(const std::string & data) : data(data) {}
    void read(void * dst, size_t size) override {
        require(size <= data.size() - offset, "truncated guard fixture");
        memcpy(dst, data.data() + offset, size);
        offset += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t at, size_t size) override {
        require(size <= data.size() - offset, "truncated guard tensor");
        ggml_backend_tensor_set(tensor, data.data() + offset, at, size);
        offset += size;
    }
    size_t n_bytes() override { return offset; }
private:
    const std::string & data;
    size_t offset = 8; // Skip the host sequence magic and source sequence ID.
};

class state_writer : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override { data.append((const char *) src, size); }
    void write_tensor(ggml_tensor * tensor, size_t at, size_t size) override {
        const size_t offset = data.size();
        data.resize(offset + size);
        ggml_backend_tensor_get(tensor, &data[offset], at, size);
    }
    size_t n_bytes() override { return data.size(); }
    std::string data;
};

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open guard fixture");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: llama-hybrid-check-guards MODEL.gguf GOLDEN_STEM");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        hybrid::model_ptr model(llama_model_load_from_file(argv[1], params));
        require(bool(model), "cannot load model");
        const auto metadata = hybrid::parse_json(read_file(std::string(argv[2]) + ".tokens.json"));
        const auto prefix = read_file(std::string(argv[2]) + ".prefix.state");
        const auto delta = hybrid::read_delta(read_file(std::string(argv[2]) + ".delta"),
                                              hybrid::file_hash(argv[1]), metadata.at("n0").get<int>());
        // Real allocated KV caches use the supplied model's layout. This checks rejection, not SWA inference.
        for (auto type : {LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD, LLAMA_SWA_TYPE_CHUNKED, LLAMA_SWA_TYPE_SYMMETRIC}) {
            llama_kv_cache cache(*model, false, model->hparams, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                 false, false, false, 1024, 2, 256, type, nullptr,
                                 [&](int32_t il) { return !model->hparams.is_recr(il); }, nullptr, nullptr);
            for (int seq = 0; seq < 2; ++seq) {
                state_reader input(prefix);
                cache.state_read(input, seq, 0);
            }
            state_writer before;
            cache.state_write(before, -1, 0);
            state_reader input(delta.state);
            if (type == LLAMA_SWA_TYPE_NONE) {
                cache.state_read(input, 0, LLAMA_STATE_SEQ_FLAGS_APPEND);
                require(cache.seq_pos_max(0) + 1 == metadata.at("n1").get<int>(), "non-SWA append rejected");
                std::cout << "PASS non-SWA append positive control\n";
                continue;
            }
            bool rejected = false;
            try {
                cache.state_read(input, 0, LLAMA_STATE_SEQ_FLAGS_APPEND);
            } catch (const std::runtime_error & error) {
                require(std::string(error.what()) == "append of sliding-window state is unsupported", "wrong SWA error");
                rejected = true;
            }
            require(rejected, "SWA append accepted");
            state_writer after;
            cache.state_write(after, -1, 0);
            require(after.data == before.data, "SWA rejection changed KV bytes or metadata");
            std::cout << "PASS SWA guard type=" << type << " both sequences byte-identical\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
