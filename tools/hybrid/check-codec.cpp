// Codec controls retained from A80013538; synthetic native body does not test engine import.
#include "hybrid-state.h"
#include "hash/hash.h"
#include "hash/xxhash/xxhash.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
using hybrid::json;
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static json header(const std::string & hash, size_t n0, size_t n1, const std::vector<llama_token> & tokens) {
    return {{"model_sha256", hash}, {"session_id", "golden"}, {"n0", n0}, {"n1", n1}, {"tokens", std::vector<llama_token>(tokens.begin() + n0, tokens.begin() + n1)}, {"prefix_hash_n1", hybrid::prefix_hash(tokens)}};
}

static void bf16_rounding_tests() {
    // A tiny native-shaped fixture checks ties, normal finite error, signed zero, NaN, Inf, underflow and overflow.
    std::string raw;
    auto add = [&](uint64_t x, size_t n) { for (size_t i = 0; i < n; ++i) { raw += char(x >> (8 * i)); } };
    add(0, 8); add(1, 4); add(1, 4); add(0, 4); add(1, 4); raw.append(16, char(0));
    add(0, 4); add(1, 4); add(0, 8);
    for (int i = 0; i < 2; ++i) { add(GGML_TYPE_Q8_0, 4); add(34, 8); raw.append(34, char(17)); }
    add(1, 4); add(0, 4); add(0, 4); add(0, 4); add(2, 4);
    std::vector<uint32_t> values = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7f800001, 0xffc01234,
        0x3f808000, 0x3f818000, 0x00800000, 1, 0x7f7fffff};
    for (uint32_t exponent = 1; exponent < 255; ++exponent) {
        for (uint32_t mantissa : {0u, 0x7fffu, 0x8000u, 0xffffu, 0x3f8000u, 0x7f7fffu}) {
            values.push_back((exponent << 23) | mantissa);
            values.push_back(0x80000000u | (exponent << 23) | mantissa);
        }
    }
    std::vector<size_t> offsets;
    for (int i = 0; i < 2; ++i) {
        add(GGML_TYPE_F32, 4); add(values.size() * 4, 8); offsets.push_back(raw.size());
        for (auto value : values) { add(value, 4); }
    }
    const auto packed = hybrid::transcode_recurrent(raw, true, true);
    const auto back = hybrid::transcode_recurrent(packed, false, true);
    require(raw.size() - packed.size() == values.size() * 4 && back.size() == raw.size(), "bf16 payload size");
    double maximum = 0;
    for (auto offset : offsets) {
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t actual; std::memcpy(&actual, back.data() + offset + i * 4, 4);
            const uint32_t bits = values[i];
            float original, rounded; std::memcpy(&original, &bits, 4); std::memcpy(&rounded, &actual, 4);
            if (std::isnan(original)) { require(std::isnan(rounded), "bf16 NaN became non-NaN"); }
            else if (i == 6) { require(actual == 0x3f800000, "bf16 even tie rounded up"); }
            else if (i == 7) { require(actual == 0x3f820000, "bf16 odd tie rounded down"); }
            else if (i == 9) { require(actual == 0, "bf16 minimum subnormal did not underflow"); }
            else if (i == 10) { require(actual == 0x7f800000, "bf16 finite overflow not infinity"); }
            else if (!std::isnormal(original)) { require(actual == bits, "bf16 Inf or signed zero changed"); }
            if (std::isnormal(original) && std::isfinite(rounded)) {
                const double relative = std::abs((double(rounded) - original) / original);
                maximum = std::max(maximum, relative);
                require(relative <= std::ldexp(1., -8), "bf16 relative error exceeds 2^-8");
            }
        }
    }
    std::cout << "PASS test_bf16_rne_normal_bound_nan_inf max_relative_error=" << maximum << '\n';
}

static void wire_size_tests() {
    // Layout arithmetic, without a 27B allocation, accounts for exactly three omitted native fixed-state records.
    const uint64_t kv_row = 16 * 4 * (256 + 256) * 34 / 32;
    const uint64_t recurrent = 48ull * 4 * (128 * 6144 + 3 * (6144 + 2 * 16 * 128));
    const uint64_t fixed_record = recurrent + 12 + 8 + 2 * 48 * 12;
    const uint64_t kv_header = 8 + 8 + 16 + 2 * 16 * 12;
    const int count = 40000, width = 10871;
    const int chunks = (count + width - 1) / width;
    uint64_t old_wire = 0, new_wire = 0, old_body = 0, new_body = 0;
    const std::vector<llama_token> synthetic(count, 100);
    for (int i = 0, n0 = 0; n0 < count; ++i, n0 += width) {
        const int n1 = std::min(count, n0 + width);
        auto h = header(std::string(64, 'a'), n0, n1, synthetic);
        h["session_id"] = std::string(32, 'a');
        const uint64_t kv = kv_header + (kv_row + 24) * (n1 - n0);
        const uint64_t before = kv + fixed_record, after = kv + (i + 1 == chunks ? fixed_record : 0);
        auto legacy = h;
        legacy["magic"] = "HYBD";
        legacy["version"] = 1;
        legacy["body_format"] = "llama-native-seq-v1";
        legacy["kv_type"] = "q8_0";
        legacy["state_dtype"] = "f32";
        legacy["sections"] = json::array({{{"name", "llama_seq_state"}, {"offset", 0}, {"bytes", before}}});
        h["transfer_id"] = std::string(32, 'b');
        h["transfer_n0"] = 0; h["transfer_n1"] = count;
        h["chunk_index"] = i; h["chunk_count"] = chunks; h["fixed_state"] = i + 1 == chunks;
        const auto text = hybrid::encode_delta_header(h, after);
        require(after <= hybrid::max_delta_body_bytes && before <= hybrid::max_delta_body_bytes, "synthetic body over cap");
        old_wire += 4 + legacy.dump().size() + before + 48;
        new_wire += 4 + text.size() + after + 48;
        old_body += before; new_body += after;
    }
    require(kv_row == 34816 && recurrent == 156893184 && chunks == 4, "wrong 27B geometry");
    require(old_body - new_body == 3 * fixed_record, "three fixed records not removed");
    std::cout << "PASS wire_size_27b_40000 old_body=" << old_body << " new_body=" << new_body
              << " old_wire=" << old_wire << " new_wire=" << new_wire << " saved_wire=" << old_wire - new_wire
              << " recurrent_tensor=" << recurrent << " fixed_record=" << fixed_record << '\n';

}

static void codec_tests(size_t max_ctx) {
    // Round trips accept the valid control; corrupt, truncated and incompatible envelopes are rejected before engine mutation.
    const json layout = {{"architecture", "qwen35"}, {"layer_count", 24}, {"embedding_size", 1024},
                         {"heads", 8}, {"kv_heads", 2}, {"recurrent_layout", json::array({true, false})},
                         {"vocab_sha256", std::string(64, 'a')}};
    const auto fingerprint = hybrid::layout_fingerprint(layout);
    json reordered = json::object();
    for (const auto & field : {"vocab_sha256", "recurrent_layout", "kv_heads", "heads", "embedding_size", "layer_count", "architecture"}) {
        reordered[field] = layout.at(field);
    }
    require(hybrid::layout_fingerprint(reordered) == fingerprint, "layout fingerprint depends on insertion order");
    for (const auto & field : {"architecture", "layer_count", "embedding_size", "heads", "kv_heads", "recurrent_layout", "vocab_sha256"}) {
        auto changed = layout;
        changed[field] = "changed";
        require(hybrid::layout_fingerprint(changed) != fingerprint, "different layout fingerprint matched");
    }
    require(hybrid::layout_fingerprint(layout) == fingerprint, "layout fingerprint unstable");
    std::cout << "PASS layout fingerprint structural mutation controls\n";
    wire_size_tests();
    bf16_rounding_tests();
    const std::string hash(64, 'a');
    auto h = header(hash, 0, 1, {1});
    // The writer admits the inclusive body cap and rejects one byte more before allocating a body.
    hybrid::encode_delta_header(h, hybrid::max_delta_body_bytes);
    bool oversized_body_rejected = false;
    try { hybrid::encode_delta_header(h, hybrid::max_delta_body_bytes + 1); }
    catch (const std::exception & e) { oversized_body_rejected = std::string(e.what()) == "invalid delta size"; }
    require(oversized_body_rejected, "one-over body accepted");
    std::cout << "PASS native body cap inclusive and one-over rejection\n";
    // The 0.8B and 27B reserved chunk maxima pass the node's size gate; one more token exceeds the bound including reserve.
    for (const auto & geometry : {std::make_pair(size_t(6528), size_t(20201472)), std::make_pair(size_t(34816), size_t(157000000))}) {
        const size_t slope = geometry.first + 24;
        const size_t fixed = geometry.second + 1024 * 1024;
        const size_t chunk = (hybrid::max_delta_body_bytes - fixed) / slope;
        hybrid::encode_delta_header(h, fixed + chunk * slope);
        bool one_more_rejected = false;
        try { hybrid::encode_delta_header(h, fixed + (chunk + 1) * slope); }
        catch (const std::exception & e) { one_more_rejected = std::string(e.what()) == "invalid delta size"; }
        require(one_more_rejected, "one-more reserved chunk accepted");
        std::cout << "PASS reserved chunk boundary: kv=" << geometry.first << " tokens=" << chunk << '\n';
    }
    // Frozen pre-bf16 envelope fields and order protect the default wire, independent of the decoder.
    const std::string legacy_header = R"({"model_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","session_id":"golden","n0":0,"n1":1,"tokens":[1],"prefix_hash_n1":"67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450","magic":"HYBD","version":2,"transfer_id":"golden","transfer_n0":0,"transfer_n1":1,"chunk_index":0,"chunk_count":1,"fixed_state":true,"body_format":"llama-native-chunks-v3","kv_type":"q8_0","state_dtype":"f32","sections":[{"name":"llama_seq_state","offset":0,"bytes":3}]})";
    require(hybrid::encode_delta_header(h, 3) == legacy_header, "f32 legacy header bytes changed");
    auto wire = hybrid::encode_delta(h, "abc");
    require(hybrid::read_delta(wire, hash, 0).state == "abc", "codec positive control");
    // Legacy exact harness admission accepts f32 and refuses labelled bf16 before native expansion.
    require(hybrid::read_delta(wire, hash, 0, "f32").state == "abc", "f32 harness control");
    auto dtype_header = hybrid::parse_json(legacy_header);
    dtype_header["state_dtype"] = "bf16";
    const auto dtype_text = dtype_header.dump();
    std::string dtype_wire;
    for (int i = 0; i < 4; ++i) { dtype_wire += char(dtype_text.size() >> (8 * i)); }
    dtype_wire += dtype_text + wire.substr(4 + legacy_header.size());
    bool mismatch = false;
    try { hybrid::read_delta(dtype_wire, hash, 0, "f32"); }
    catch (const std::exception & e) { mismatch = std::string(e.what()) == "state dtype mismatch"; }
    require(mismatch, "legacy exact harness expanded bf16");
    std::cout << "PASS test_legacy_harness_dtype_admission_with_f32_control\n";
    // Zero-token transfers are outside HYBD for both dtypes; valid f32 remains byte-identical.
    for (const auto * dtype : {"f32", "bf16"}) {
        auto empty = header(hash, 0, 0, {}); empty["state_dtype"] = dtype;
        bool refused = false;
        try { hybrid::encode_delta(empty, "abc"); }
        catch (const std::exception & e) { refused = std::string(e.what()) == "invalid positions"; }
        require(refused, "empty delta dtype-specific refusal");
        auto nonempty = h; nonempty["state_dtype"] = dtype;
        refused = false;
        try { hybrid::encode_delta(nonempty, ""); }
        catch (const std::exception & e) { refused = std::string(e.what()) == "invalid delta size"; }
        require(refused, "empty body dtype-specific refusal");
    }
    require(hybrid::encode_delta(h, "abc") == wire, "valid f32 control changed");
    std::cout << "PASS test_empty_delta_shared_refusal_with_nonempty_control\n";
    // A borrowed decode references the owning envelope; the copying API retains its independent state.
    const auto view = hybrid::read_delta_view(wire, hash, 0);
    const auto owned = hybrid::read_delta(wire, hash, 0);
    require(view.state == owned.state && view.header == owned.header, "borrowed decode differs");
    require(view.state.data() == wire.data() + wire.size() - 48 - 3, "borrowed decode copied the body");
    require(owned.state.data() != view.state.data(), "owning decode borrowed the body");
    std::cout << "PASS test_borrowed_delta_lifetime_and_owning_control\n";
    // Streaming checksums match the independent portable SHA and one-shot XXH3 across block/chunk boundaries and binary tails.
    std::vector<size_t> sizes = {65535, 65536, 65537, 131135};
    for (size_t i = 1; i <= 130; ++i) { sizes.push_back(i); }
    for (size_t size : sizes) {
        std::string body(size, '\0');
        for (size_t i = 0; i < size; ++i) { body[i] = (char) (i * 37 + i / 17); }
        const auto encoded = hybrid::encode_delta(h, body);
        const auto digest = hash_sha256_hex(body.data(), body.size());
        static const char digits[] = "0123456789abcdef";
        std::string actual;
        for (size_t i = encoded.size() - 32; i < encoded.size(); ++i) {
            const auto c = (unsigned char) encoded[i];
            actual += digits[c >> 4]; actual += digits[c & 15];
        }
        XXH128_canonical_t xxh;
        XXH128_canonicalFromHash(&xxh, XXH3_128bits(body.data(), body.size()));
        require(digest == actual, "test_streaming_checksum_block_chunk_and_binary_boundaries: SHA-256 differs from portable");
        require(std::memcmp(&xxh, encoded.data() + encoded.size() - 48, 16) == 0, "streaming XXH3 differs from one-shot");
        require(hybrid::read_delta(encoded, hash, 0).state == body, "binary checksum round trip failed");
    }
    require(hash_sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 known answer");
    std::cout << "PASS test_streaming_checksum_block_chunk_and_binary_boundaries\n";
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
    corrupt = wire; corrupt[corrupt.size() - 48] ^= 1;
    rejects("corrupt-section-hash", corrupt, hash, 0, "checksum mismatch");
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
    // HYBD 2 from a chunk-only build has the old body format and must refuse before native parsing.
    bad_h = hybrid::parse_json(wire.substr(4, length));
    bad_h["body_format"] = "llama-native-chunks-v2";
    const auto old_text = bad_h.dump();
    bad.clear();
    for (int i = 0; i < 4; ++i) { bad += (char) (old_text.size() >> (8 * i)); }
    bad += old_text + wire.substr(4 + length);
    rejects("chunk-only-body-format", bad, hash, 0, "unsupported delta layout");
    auto wrong_tokens = h;
    wrong_tokens["tokens"] = json::array();
    rejects("token-count", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "token count mismatch");
    wrong_tokens["tokens"] = json::array({-1});
    rejects("negative-token", hybrid::encode_delta(wrong_tokens, "abc"), hash, 0, "invalid delta token");
    auto wrong_prefix = h;
    wrong_prefix["prefix_hash_n1"] = std::string(64, '0');
    rejects("prefix-hash", hybrid::encode_delta(wrong_prefix, "abc"), hash, 0, "prefix hash mismatch");
    // Worst-case token IDs fit a bounded chunk of the configured context; exact-budget accepts and one-over rejects.
    const size_t count = std::min<size_t>(max_ctx, (hybrid::max_delta_header_bytes - 4096) / 11 + 1);
    std::vector<llama_token> maximum_tokens(count, INT32_MAX);
    auto maximum = header(hash, 0, maximum_tokens.size(), maximum_tokens);
    auto maximum_text = hybrid::encode_delta_header(maximum, hybrid::max_delta_body_bytes);
    require(maximum_text.size() <= std::min(hybrid::max_delta_header_bytes, 4096 + 11 * count), "maximum header budget");
    auto maximum_delta = hybrid::read_delta(hybrid::encode_delta(maximum, "abc"), hash, 0);
    require(maximum_delta.state == "abc" && maximum_delta.header.at("tokens").get<std::vector<llama_token>>() == maximum_tokens, "maximum header round trip");
    const size_t budget = std::min(hybrid::max_delta_header_bytes, 4096 + 11 * count);
    maximum["padding"] = "";
    const size_t unpadded = hybrid::encode_delta_header(maximum, 3).size();
    maximum["padding"] = std::string(budget - unpadded, 'x');
    auto exact_wire = hybrid::encode_delta(maximum, "abc");
    require(hybrid::read_delta(exact_wire, hash, 0).header.at("tokens").size() == count, "exact budget rejected");
    maximum["padding"].get_ref<std::string &>() += 'x';
    bool one_over_rejected = false;
    try { hybrid::encode_delta_header(maximum, 3); }
    catch (const std::exception & e) { one_over_rejected = std::string(e.what()) == "delta header exceeds range budget"; }
    require(one_over_rejected, "one-over writer accepted");
    uint32_t exact_size = 0;
    for (int i = 0; i < 4; ++i) { exact_size |= (uint32_t) (unsigned char) exact_wire[i] << (8 * i); }
    auto over_header = hybrid::parse_json(exact_wire.substr(4, exact_size));
    over_header["padding"].get_ref<std::string &>() += 'x';
    const auto over_text = over_header.dump();
    std::string over_wire;
    for (int i = 0; i < 4; ++i) { over_wire += (char) (over_text.size() >> (8 * i)); }
    over_wire += over_text + exact_wire.substr(4 + exact_size);
    rejects("one-over-reader", over_wire, hash, 0, budget == hybrid::max_delta_header_bytes ? "invalid delta header length" : "delta header exceeds range budget");
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
    std::cout << "PASS test_header_budget_int32_max_ctx=" << max_ctx << " chunk=" << count << " exact_budget=" << budget << ": " << maximum_text.size() << " bytes with largest body metadata; writer/reader over-budget rejection\n";
    std::cout << "PASS codec positive and negative controls\n";
}

int main(int argc, char ** argv) {
    try {
        size_t max_ctx = 16384;
        if (argc == 3 && std::string(argv[1]) == "--max-ctx") { max_ctx = std::stoul(argv[2]); }
        else { require(argc == 1, "usage: llama-hybrid-check-codec [--max-ctx N]"); }
        require(max_ctx >= 256 && max_ctx <= INT32_MAX - 256, "invalid max-ctx");
        codec_tests(max_ctx);
        return 0;
    }
    catch (const std::exception & error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
