#include "hybrid-state.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 4 || (argc == 4 && std::string(argv[2]) != "--file-chunk")) {
            throw std::runtime_error("usage: llama-hybrid-check-hash FILE [UPDATE_BYTES | --file-chunk READ_BYTES]");
        }
        const auto start = std::chrono::steady_clock::now();
        std::string digest;
        const char * backend = "portable-stream";
        size_t chunk = 65536;
        if (argc >= 3) {
            const std::string argument = argv[argc - 1];
            if (argument.empty() || argument.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("invalid update size");
            }
            chunk = std::stoull(argument);
            if (chunk == 0 || chunk > (argc == 4 ? 65536 : 1024 * 1024)) {
                throw std::runtime_error("invalid update size");
            }
        }
        if (argc == 2 || argc == 4) {
            digest = hybrid::file_hash(argv[1], chunk, &backend);
        } else {
            std::ifstream file(argv[1], std::ios::binary);
            if (!file.good()) {
                throw std::runtime_error("cannot open fixture");
            }
            // Offset input also exercises block loads without word alignment.
            std::vector<unsigned char> buffer(chunk + 1);
            auto * data = buffer.data() + 1;
            sha256_t hash;
            sha256_init(&hash);
            sha256_update(&hash, nullptr, 0);
            while (file.read(reinterpret_cast<char *>(data), chunk) || file.gcount()) {
                sha256_update(&hash, data, file.gcount());
                sha256_update(&hash, nullptr, 0);
            }
            if (!file.eof()) {
                throw std::runtime_error("cannot read fixture");
            }
            unsigned char raw[32];
            sha256_final(&hash, raw);
            static const char hex[] = "0123456789abcdef";
            for (auto byte : raw) {
                digest += hex[byte >> 4];
                digest += hex[byte & 15];
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto bytes = std::filesystem::file_size(argv[1]);
        std::cout << hybrid::json({{"sha256", digest}, {"bytes", bytes}, {"seconds", seconds},
                                  {"MB_per_second", bytes / seconds / 1000000.0}, {"backend", backend}}).dump() << '\n';
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
