#include "m4/safetensors.hpp"
#include "m4/json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace m4 {

const char* dtype_name(DType d) {
    switch (d) { case DType::F32: return "F32"; case DType::F16: return "F16"; case DType::BF16: return "BF16";
                 case DType::F8E4M3: return "F8_E4M3"; case DType::I32: return "I32"; default: return "?"; }
}
static DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "F8_E4M3") return DType::F8E4M3;
    if (s == "I32") return DType::I32;
    return DType::Unknown;
}

StFiles::~StFiles() {
    for (auto& m : maps_) { if (m.base) munmap(m.base, m.size); if (m.fd >= 0) close(m.fd); }
}

bool StFiles::open(const std::vector<std::string>& paths, std::string& err) {
    for (size_t si = 0; si < paths.size(); ++si) {
        const std::string& p = paths[si];
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open " + p; return false; }
        struct stat st; fstat(fd, &st);
        uint64_t fsz = (uint64_t) st.st_size, hlen = 0;
        if (fsz < 8 || pread(fd, &hlen, 8, 0) != 8) { err = p + ": too short"; close(fd); return false; }
        if (hlen == 0 || 8 + hlen > fsz || hlen > (1ull << 31)) { err = p + ": bad header length"; close(fd); return false; }
        std::string hdr(hlen, '\0');
        if ((uint64_t) pread(fd, &hdr[0], hlen, 8) != hlen) { err = p + ": short header read"; close(fd); return false; }
        Json j; std::string jerr;
        if (!parse_json(hdr, j, jerr) || j.t != Json::Obj) { err = p + ": header JSON: " + jerr; close(fd); return false; }
        void* base = mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { err = p + ": mmap failed"; close(fd); return false; }
        maps_.push_back({base, fsz, fd});
        const uint8_t* data0 = (const uint8_t*) base + 8 + hlen;
        const uint64_t dsz = fsz - 8 - hlen;
        for (const auto& kv : j.o) {
            if (kv.first == "__metadata__") continue;
            StTensor t; t.name = kv.first; t.shard = (int) si;
            t.dtype_str = kv.second.str("dtype"); t.dtype = parse_dtype(t.dtype_str);
            const Json* sh = kv.second.get("shape"); const Json* of = kv.second.get("data_offsets");
            if (!sh || !of || of->a.size() != 2) { err = p + ": tensor " + t.name + " malformed"; return false; }
            for (const auto& d : sh->a) t.shape.push_back((int64_t) d.n);
            t.begin = (uint64_t) of->a[0].n; t.end = (uint64_t) of->a[1].n;
            if (t.end < t.begin || t.end > dsz) { err = p + ": tensor " + t.name + " lies outside the file"; return false; }
            if (t.dtype != DType::Unknown && (uint64_t) t.numel() * dtype_bytes(t.dtype) != t.nbytes()) {
                err = p + ": tensor " + t.name + " size does not match shape x dtype"; return false;
            }
            t.data = data0 + t.begin;
            if (by_name_.count(t.name)) { err = "tensor " + t.name + " appears in two shards"; return false; }
            by_name_[t.name] = tensors_.size();
            tensors_.push_back(std::move(t));
        }
    }
    return true;
}

const StTensor* StFiles::find(const std::string& n) const {
    auto it = by_name_.find(n);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}
uint64_t StFiles::total_bytes() const { uint64_t s = 0; for (auto& t : tensors_) s += t.nbytes(); return s; }
void StFiles::advise_random(bool on) { for (auto& m : maps_) madvise(m.base, m.size, on ? MADV_RANDOM : MADV_NORMAL); }

bool discover_shards(const std::string& path, std::vector<std::string>& out, std::string& err) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) { err = "no such path: " + path; return false; }
    if (S_ISREG(st.st_mode)) { out = {path}; return true; }
    DIR* d = opendir(path.c_str());
    if (!d) { err = "cannot read directory " + path; return false; }
    std::vector<std::string> model, other;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() < 12 || n.compare(n.size() - 12, 12, ".safetensors") != 0) continue;
        if (n.rfind("model-", 0) == 0) model.push_back(path + "/" + n);
        else if (n.rfind("consolidated", 0) != 0) other.push_back(path + "/" + n);
    }
    closedir(d);
    out = !model.empty() ? model : other;     // the repo also carries "consolidated-*" (Mistral-native names): not used
    std::sort(out.begin(), out.end());
    if (out.empty()) { err = "no model-*.safetensors in " + path; return false; }
    return true;
}

}  // namespace m4
