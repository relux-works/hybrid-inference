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
        int n0 = 0, committed = 0;
        std::vector<std::unique_ptr<std::string>> wires;
        std::vector<hybrid::delta_view> pending;
        for (int arg = 1; arg < argc - 1; ++arg) {
            wires.push_back(std::make_unique<std::string>(read_file(argv[arg])));
            auto delta = hybrid::read_delta_view(*wires.back(), target.at("model_sha256"), n0);
            require(delta.state.data() == wires.back()->data() + wires.back()->size() - 48 - delta.state.size(), "borrowed reader copied native body");
            require(delta.header.at("layout_fingerprint") == hybrid::layout_fingerprint(target.at("model_layout")), "native fingerprint mismatch");
            const int n1 = delta.header.at("n1");
            const bool final = delta.header.at("fixed_state");
            require(n1 > n0 && n1 <= 1024, "synthetic native check limited to 1024 tokens");
            require(delta.header.at("chunk_index") == pending.size(), "chunk order mismatch");
            if (pending.empty()) {
                require(delta.header.at("transfer_n0") == committed, "transfer start mismatch");
            } else {
                for (const auto * key : {"session_id", "transfer_id", "transfer_n0", "transfer_n1", "chunk_count"}) {
                    require(delta.header.at(key) == pending.front().header.at(key), "transfer identity mismatch");
                }
            }
            const llama_state_seq_chunk chunk {reinterpret_cast<const uint8_t *>(delta.state.data()), delta.state.size(), n1};
            require(llama_memory_seq_read_chunks(&memory, hp, {chunk}, 0, n0, 1024, true, final) == delta.state.size(), "invalid native chunk");
            require(memory.seq_pos_max(0) == committed - 1 && memory.get_mem_attn()->seq_pos_max(0) == committed - 1,
                    "staged chunk changed memory");
            pending.push_back(std::move(delta));
            if (final) {
                std::vector<llama_state_seq_chunk> chunks;
                size_t bytes = 0;
                for (const auto & part : pending) {
                    chunks.push_back({reinterpret_cast<const uint8_t *>(part.state.data()), part.state.size(), part.header.at("n1")});
                    bytes += part.state.size();
                }
                require(llama_memory_seq_read_chunks(&memory, hp, chunks, 0, committed, 1024, false, true) == bytes, "native chunk import failed");
                require(memory.seq_pos_max(0) == n1 - 1, "native endpoint mismatch");
                for (const auto & part : pending) {
                    writer output;
                    if (part.header.at("fixed_state").get<bool>()) {
                        memory.state_write_range(output, 0, part.header.at("n0"), part.header.at("n1"), 0);
                    } else {
                        memory.get_mem_attn()->state_write_range(output, 0, part.header.at("n0"), part.header.at("n1"), 0);
                    }
                    require(std::string_view(output.bytes) == part.state.substr(8), "native round-trip byte mismatch");
                }
                committed = n1;
                pending.clear();
                wires.clear();
            }
            n0 = n1;
        }
        require(pending.empty(), "missing final chunk");
        std::cout << "PASS native envelope, fingerprint, attention/GDN cache import and byte round-trip; borrowed native body views\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
