// Codec controls retained from A80013538; synthetic native body does not test engine import.
#include "hybrid-state.h"
#include <iostream>
#include <stdexcept>
using hybrid::json;
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static json header(const std::string & hash, size_t n0, size_t n1, const std::vector<llama_token> & tokens) {
    return {{"model_sha256", hash}, {"session_id", "golden"}, {"n0", n0}, {"n1", n1}, {"tokens", std::vector<llama_token>(tokens.begin() + n0, tokens.begin() + n1)}, {"prefix_hash_n1", hybrid::prefix_hash(tokens)}};
}

static void codec_tests() {
    // Round trips accept the valid control; corrupt, truncated and incompatible envelopes are rejected before engine mutation.
    const std::string hash(64, 'a');
    auto h = header(hash, 0, 1, {1});
    auto wire = hybrid::encode_delta(h, "abc");
    require(hybrid::read_delta(wire, hash, 0).state == "abc", "codec positive control");
    require(hybrid::prefix_hash({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty prefix hash");
    auto rejects = [&](const std::string & name, const std::string & input, const std::string & model_hash, int64_t n0, const std::string & expected) {
        try { hybrid::read_delta(input, model_hash, n0); }
        catch (const std::exception & e) {
            require(std::string(e.what()).find(expected) != std::string::npos, "wrong rejection reason");
            std::cout << "PASS " << name << ": " << e.what() << '\n';
            return;
        }
        throw std::runtime_error("accepted " + name);
    };
    rejects("wrong-model", wire, std::string(64, 'b'), 0, "model hash mismatch");
    rejects("overlap", wire, hash, 1, "position gap or overlap");
    auto gap_h = h; gap_h["n0"] = 2; gap_h["n1"] = 3;
    rejects("gap", hybrid::encode_delta(gap_h, "abc"), hash, 0, "position gap or overlap");
    rejects("truncated", wire.substr(0, wire.size() - 1), hash, 0, "truncated or trailing");
    rejects("trailing", wire + "x", hash, 0, "truncated or trailing");
    auto corrupt = wire; corrupt[corrupt.size() - 49] ^= 1;
    rejects("corrupt-section", corrupt, hash, 0, "checksum mismatch");
    corrupt = wire; corrupt[corrupt.size() - 1] ^= 1;
    rejects("corrupt-body-hash", corrupt, hash, 0, "checksum mismatch");
    uint32_t length = 0;
    for (int i = 0; i < 4; ++i) { length |= (uint32_t) (unsigned char) wire[i] << (8 * i); }
    auto bad_h = hybrid::parse_json(wire.substr(4, length));
    bad_h["kv_type"] = "f16";
    const std::string text = bad_h.dump();
    std::string bad;
    for (int i = 0; i < 4; ++i) { bad += (char) (text.size() >> (8 * i)); }
    bad += text + wire.substr(4 + length);
    rejects("layout", bad, hash, 0, "unsupported delta layout");
    auto wrong_tokens = h;
    wrong_tokens["tokens"] = json::array();
    rejects("token-count", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "token count mismatch");
    wrong_tokens["tokens"] = json::array({-1});
    rejects("negative-token", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "invalid delta token");
    auto wrong_prefix = h;
    wrong_prefix["prefix_hash_n1"] = std::string(64, '0');
    rejects("prefix-hash", hybrid::encode_delta(wrong_prefix, "abc"), hash, 0, "prefix hash mismatch");
    // The maximum node range fits even with ten-digit int32 IDs; both readers and writers enforce the range budget.
    std::vector<llama_token> maximum_tokens(16384, INT32_MAX);
    auto maximum = header(hash, 0, maximum_tokens.size(), maximum_tokens);
    auto maximum_text = hybrid::encode_delta_header(maximum, hybrid::max_delta_body_bytes);
    require(maximum_text.size() > 65536 && maximum_text.size() <= 4096 + 11 * 16384, "maximum header budget");
    auto maximum_delta = hybrid::read_delta(hybrid::encode_delta(maximum, "abc"), hash, 0);
    require(maximum_delta.state == "abc" && maximum_delta.header.at("tokens").get<std::vector<llama_token>>() == maximum_tokens, "maximum header round trip");
    auto padded = h;
    padded["padding"] = std::string(4096, 'x');
    bool rejected = false;
    try { hybrid::encode_delta_header(padded, hybrid::max_delta_body_bytes); }
    catch (const std::exception & e) { rejected = std::string(e.what()) == "delta header exceeds range budget"; }
    require(rejected, "writer header budget not enforced");
    padded = hybrid::parse_json(wire.substr(4, length));
    padded["padding"] = std::string(4096, 'x');
    const auto padded_text = padded.dump();
    std::string padded_wire;
    for (int i = 0; i < 4; ++i) { padded_wire += (char) (padded_text.size() >> (8 * i)); }
    padded_wire += padded_text + wire.substr(4 + length);
    rejects("reader-range-header-budget", padded_wire, hash, 0, "delta header exceeds range budget");
    std::string oversized_length;
    for (int i = 0; i < 4; ++i) { oversized_length += (char) ((1048577u >> (8 * i)) & 255); }
    rejects("reader-absolute-header-budget", oversized_length, hash, 0, "invalid delta header length");
    std::cout << "PASS test_header_budget_int32_max_16384: " << maximum_text.size() << " bytes with largest body metadata; writer/reader over-budget rejection\n";
    std::cout << "PASS codec positive and negative controls\n";
}

int main() {
    try { codec_tests(); return 0; }
    catch (const std::exception & error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
