// strata_dump.cpp - EXPERIMENTAL (2026-10-07): read a Strata session file (STRSESS v1) with Strata's own
// session_file_read and dump every array as raw little-endian files plus meta.json, for the Strata -> Sushi
// prefix-cache converter.  Build (on the GPU box):
//   g++ -std=c++20 -O2 -I ~/strata/include strata_dump.cpp ~/strata/src/core/conversation_file.cpp -o strata_dump
// Usage: strata_dump SESSION_FILE OUT_DIR
#include "strata/core/conversation_file.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::core;

static bool put(const std::string& path, const void* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (n) f.write(static_cast<const char*>(p), (std::streamsize) n);
    return (bool) f;
}
static bool put_buf(const std::string& path, const ConversationBuffer& b) {
    std::ofstream f(path, std::ios::binary);
    bool ok = b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) {
        f.write(reinterpret_cast<const char*>(p), (std::streamsize) n);
        return (bool) f;
    });
    return ok && (bool) f;
}
static std::string cp_json(const ConversationCheckpoint& c) {
    char s[256];
    std::snprintf(s, sizeof s, "{\"tokens\":%zu,\"images\":%zu,\"gdn\":%zu,\"ple\":%zu,\"tails\":%zu,\"dead\":%zu,\"block_pos\":%zu}",
                  c.ids.size(), c.imgs.size(), c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size());
    return s;
}
static void dump_cp(const std::string& dir, const std::string& tag, const ConversationCheckpoint& c) {
    put(dir + "/" + tag + "_ids.i32", c.ids.data(), c.ids.size() * sizeof(int32_t));
    put(dir + "/" + tag + "_gdn.f32", c.gdn.data(), c.gdn.size());
    put(dir + "/" + tag + "_ple.f32", c.ple.data(), c.ple.size());
    put(dir + "/" + tag + "_tails.f32", c.tails.data(), c.tails.size());
    put(dir + "/" + tag + "_dead.f32", c.dead.data(), c.dead.size());
    put(dir + "/" + tag + "_block_pos.i32", c.block_pos.data(), c.block_pos.size());
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s SESSION_FILE OUT_DIR\n", argv[0]); return 2; }
    const std::string path = argv[1], dir = argv[2];
    // identity from the header itself: magic(8) version(4) header_size(4) model_fp(8) config_fp(8)
    unsigned char h[64] = {};
    { std::ifstream f(path, std::ios::binary); f.read(reinterpret_cast<char*>(h), sizeof h);
      if (f.gcount() != 64 || std::memcmp(h, "STRSESS\x01", 8) != 0) { std::fprintf(stderr, "not a STRSESS file\n"); return 1; } }
    SessionFileIdentity id;
    std::memcpy(&id.model, h + 16, 8);
    std::memcpy(&id.config, h + 24, 8);
    SavedConversation img;
    size_t bytes = 0;
    std::string err;
    if (!session_file_read(path, id, img, bytes, err)) { std::fprintf(stderr, "read failed: %s\n", err.c_str()); return 1; }

    dump_cp(dir, "live", img.live);
    for (size_t i = 0; i < img.checkpoints.size(); ++i) dump_cp(dir, "cp" + std::to_string(i), img.checkpoints[i]);
    std::string kvj;
    for (size_t l = 0; l < img.kv.size(); ++l) {
        const auto& kv = img.kv[l];
        const std::string t = dir + "/kv" + std::to_string(l);
        put_buf(t + "_k.bin", kv.k); put_buf(t + "_v.bin", kv.v);
        put_buf(t + "_ks.bin", kv.k_scale); put_buf(t + "_vs.bin", kv.v_scale);
        put_buf(t + "_pooled.bin", kv.pooled);
        char s[320];
        std::snprintf(s, sizeof s, "%s{\"format\":%d,\"cells\":%lld,\"heads\":%lld,\"head_dim\":%lld,\"page_size\":%lld,"
                      "\"pooled_rows\":%lld,\"idx_dim\":%lld,\"k\":%zu,\"v\":%zu,\"ks\":%zu,\"vs\":%zu,\"pooled\":%zu}",
                      l ? "," : "", kv.format, (long long) kv.cells, (long long) kv.heads, (long long) kv.head_dim,
                      (long long) kv.page_size, (long long) kv.pooled_rows, (long long) kv.idx_dim, kv.k.size(),
                      kv.v.size(), kv.k_scale.size(), kv.v_scale.size(), kv.pooled.size());
        kvj += s;
    }
    std::string geo, cps;
    for (size_t i = 0; i < img.geometry.size(); ++i) geo += (i ? "," : "") + std::to_string(img.geometry[i]);
    for (size_t i = 0; i < img.checkpoints.size(); ++i) cps += (i ? "," : "") + cp_json(img.checkpoints[i]);
    std::ofstream m(dir + "/meta.json");
    m << "{\"file_bytes\":" << bytes << ",\"model_fp\":" << id.model << ",\"config_fp\":" << id.config
      << ",\"geometry\":[" << geo << "],\"layer_lo\":" << img.layer_lo << ",\"layer_hi\":" << img.layer_hi
      << ",\"cvec\":" << (img.cvec ? "true" : "false") << ",\"stage_images\":" << img.stage_images.size()
      << ",\"live\":" << cp_json(img.live) << ",\"checkpoints\":[" << cps << "],\"kv\":[" << kvj << "]}\n";
    std::printf("ok: %zu bytes, live %zu tokens, %zu checkpoints, %zu kv layers -> %s\n", bytes, img.live.ids.size(),
                img.checkpoints.size(), img.kv.size(), dir.c_str());
    return 0;
}
