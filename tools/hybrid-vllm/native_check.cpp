// Synthetic cache import checks serialization, not cross-engine numerical equivalence.
#include "hybrid-state.h"
#include "llama-ext.h"
#include "llama-io.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static std::string read_file(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good() && input.tellg() <= 513 * 1024 * 1024, "invalid check file");
    input.seekg(0);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class reader : public llama_io_read_i {
public:
    explicit reader(const std::string & bytes) : bytes(bytes) {}
    void read(void * dst, size_t n) override {
        require(n <= bytes.size() - offset, "truncated native state");
        memcpy(dst, bytes.data() + offset, n);
        offset += n;
    }
    void read_tensor(ggml_tensor * tensor, size_t at, size_t n) override {
        require(n <= bytes.size() - offset, "truncated native tensor");
        ggml_backend_tensor_set(tensor, bytes.data() + offset, at, n);
        offset += n;
    }
    size_t n_bytes() override { return offset; }
private:
    const std::string & bytes;
    size_t offset = 8;
};

class writer : public llama_io_write_i {
public:
    std::string bytes;
    void write(const void * src, size_t n) override { bytes.append((const char *) src, n); }
    void write_tensor(ggml_tensor * tensor, size_t at, size_t n) override {
        const size_t offset = bytes.size();
        bytes.resize(offset + n);
        ggml_backend_tensor_get(tensor, bytes.data() + offset, at, n);
    }
    size_t n_bytes() override { return bytes.size(); }
};

int main(int argc, char ** argv) {
    try {
        require(argc >= 3, "usage: native_check DELTA [DELTA...] TARGET.json");
        const auto target = hybrid::parse_json(read_file(argv[argc - 1]));
        llama_backend_init();
        hybrid::model_ptr model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
        auto & hp = model->hparams;
        hp = {};
        hp.n_layer_all = 64;
        hp.n_embd = 5120;
        hp.n_head_arr.fill(24);
        hp.n_head_kv_arr.fill(4);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 256;
        hp.rope_type = LLAMA_ROPE_TYPE_IMROPE;
        hp.ssm_d_conv = 4;
        hp.ssm_d_inner = 6144;
        hp.ssm_d_state = 128;
        hp.ssm_dt_rank = 48;
        hp.ssm_n_group = 16;
        for (int i = 0; i < 64; ++i) {
            hp.kv_source_impl[i] = i;
            hp.is_recr_impl[i] = (i + 1) % 4 != 0;
        }
        llama_memory_hybrid memory(*model, false, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, 1024, 0,
                                   LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, 0, false, true);
        int n0 = 0;
        for (int arg = 1; arg < argc - 1; ++arg) {
            const auto delta = hybrid::read_delta(read_file(argv[arg]), target.at("model_sha256"), n0);
            require(delta.state.size() >= 8, "truncated native sequence header");
            require(delta.header.at("layout_fingerprint") == hybrid::layout_fingerprint(target.at("model_layout")), "native fingerprint mismatch");
            uint32_t magic;
            memcpy(&magic, delta.state.data(), sizeof(magic));
            require(magic == 0xaf143cd8, "native sequence magic mismatch");
            const int n1 = delta.header.at("n1");
            require(n1 > n0 && n1 <= 1024, "synthetic native check limited to 1024 tokens");
            reader input(delta.state);
            memory.state_read(input, 0, LLAMA_STATE_SEQ_FLAGS_APPEND);
            require(input.n_bytes() == delta.state.size() && memory.seq_pos_max(0) == n1 - 1, "native endpoint mismatch");
            writer output;
            memory.state_write_range(output, 0, n0, n1, 0);
            require(output.bytes == delta.state.substr(8), "native round-trip byte mismatch");
            n0 = n1;
        }
        std::cout << "PASS native envelope, fingerprint, attention/GDN cache import and byte round-trip\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
