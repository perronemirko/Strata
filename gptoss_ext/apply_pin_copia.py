#!/usr/bin/env python3
# Usage (from the folder that contains gptoss_ext/ OR from inside gptoss_ext/):  python3 apply_pin_copia.py
import os, sys
root = 'gptoss_ext' if os.path.isdir('gptoss_ext/src') else '.'
def rd(p): return open(os.path.join(root, p)).read()
def wr(p, s): open(os.path.join(root, p), 'w').write(s)
def once(s, a, b, name):
    if s.count(a) != 1:
        sys.exit("FAILED (%s): expected 1 match, found %d of: %r" % (name, s.count(a), a[:70]))
    return s.replace(a, b)

# ---- gpu.hpp
p = 'include/gptoss/gpu.hpp'; s = rd(p)
if 'host_alloc' not in s:
    i = s.find('bool host_register(const void* p, size_t n);')
    if i < 0: sys.exit('FAILED: host_register declaration not found in gpu.hpp')
    j = s.find('\n', i) + 1
    s = s[:j] + ('void* host_alloc(size_t n);                  // page-locked host memory (cudaMallocHost); nullptr when it cannot be had. Never touches last_error()\n'
                 'void host_free(void* p);\n') + s[j:]
    wr(p, s)

# ---- gpu.cpp (emulated device)
p = 'src/gpu.cpp'; s = rd(p)
if 'host_alloc' not in s:
    i = s.find('bool host_register(const void*, size_t)')
    if i < 0: sys.exit('FAILED: host_register not found in gpu.cpp')
    j = s.find('\n', i) + 1
    s = s[:j] + 'void* host_alloc(size_t n) { return std::malloc(n ? n : 1); }\nvoid host_free(void* p) { std::free(p); }\n' + s[j:]
    wr(p, s)

# ---- gpu_cuda.cu
p = 'src/gpu_cuda.cu'; s = rd(p)
if 'host_alloc' not in s:
    end = '    } // namespace gpu\n} // namespace gptoss'
    s = once(s, end, '''
        void *host_alloc(size_t n)
        {
            void *p = nullptr;
            if (cudaMallocHost(&p, n ? n : 1) != cudaSuccess)
            {
                cudaGetLastError(); // a refusal is not a failed call of the session: no g_err
                return nullptr;
            }
            return p;
        }

        void host_free(void *p)
        {
            if (p)
                cudaFreeHost(p);
        }
''' + end, 'gpu_cuda.cu')
    wr(p, s)

# ---- model.cpp
p = 'src/model.cpp'; s = rd(p)
NEW = r'''// --pin-ram-gib: copies the expert tensors of whole layers (the layers the profile uses most first) into page-locked host
// memory and points the layer at the copy, until the budget is spent. This is the way that works on every card: registering the
// weights' file mapping in place (cudaHostRegister) needs the read-only flag that GeForce drivers often refuse. It runs at load,
// BEFORE the preload and before any request, so nothing reads the old pointers while they move. Copies out of the pinned blocks
// are direct DMA, and the CPU kernels read them as before.
void Model::Impl::ram_pin_start() {
    if (!(o.pin_ram_gib > 0) || !gpu_on) return;
    const size_t H = (size_t) c.hidden, I = (size_t) c.moe_ff, G1 = H / 32, G2 = I / 32, NE = (size_t) c.n_exp;
    const size_t sz[4] = { NE * 2 * I * G1 * 16, NE * 2 * I * G1, NE * H * G2 * 16, NE * H * G2 };
    const size_t per_layer = sz[0] + sz[1] + sz[2] + sz[3];
    const size_t budget = (size_t) (o.pin_ram_gib * 1073741824.0);
    std::vector<std::pair<uint64_t, int>> order;
    for (int l = 0; l < c.layers; ++l) {
        uint64_t u = 0;
        for (size_t e = 0; e < NE; ++e) u += usage[(size_t) l * NE + e];
        order.push_back({u, l});
    }
    std::stable_sort(order.begin(), order.end(), [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first > b.first; });
    ram_state.store(1);
    const auto t0 = std::chrono::steady_clock::now();
    size_t used = 0, n = 0;
    bool failed = false;
    for (const auto& pr : order) {
        if (used + per_layer > budget) break;
        uint8_t* buf = (uint8_t*) gpu::host_alloc(per_layer);
        if (!buf) { failed = true; break; }
        Layer& y = L[(size_t) pr.second];
        const uint8_t* src[4] = { y.gu.blocks, y.gu.scales, y.dn.blocks, y.dn.scales };
        size_t off = 0;
        for (int k = 0; k < 4; ++k) { std::memcpy(buf + off, src[k], sz[k]); off += sz[k]; }
        y.gu.blocks = buf; y.gu.scales = buf + sz[0]; y.dn.blocks = buf + sz[0] + sz[1]; y.dn.scales = buf + sz[0] + sz[1] + sz[2];
        ram_blocks.push_back(buf);
        used += per_layer; ++n;
        ram_layers.store(n); ram_bytes.store(used);
    }
    std::fprintf(stderr, "gptoss: %zu layers of experts in page-locked RAM (%.1f GiB, %.0f s)%s: their copies to the card are direct DMA\n", n, (double) used / 1073741824.0,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), failed ? "; the system would not give more pinned memory" : "");
    ram_state.store(failed && n == 0 ? 3 : 2);
}
'''
if 'ram_blocks' not in s:
    # the old background-thread version: remove ram_pin_run + the old ram_pin_start, whatever their exact text
    a = s.find('// Page-locks the four expert tensors')
    if a < 0: a = s.find('void Model::Impl::ram_pin_run() {')
    b0 = s.find('void Model::Impl::ram_pin_start() {')
    if a < 0 or b0 < 0: sys.exit('FAILED: old ram_pin_run/ram_pin_start not found in model.cpp (is punto2_pin_ram.patch applied?)')
    b1 = s.find('\n}\n', b0) + 3
    s = s[:a] + NEW + s[b1:]
    s = once(s, '    std::thread ram_th;\n', '    std::vector<void*> ram_blocks;                     // page-locked copies of whole layers\' expert tensors (freed at exit)\n', 'member')
    if '    void ram_pin_run();\n' in s: s = s.replace('    void ram_pin_run();\n', '')
    if '    if (ram_th.joinable()) ram_th.join();\n' in s: s = s.replace('    if (ram_th.joinable()) ram_th.join();\n', '')
    s = once(s, '    for (auto& sl : s.pool) gpu::release(sl.base);', '    for (void* b : s.ram_blocks) gpu::host_free(b);\n    for (auto& sl : s.pool) gpu::release(sl.base);', 'destructor')
    if '    preload_start();\n    ram_pin_start();\n' in s:
        s = s.replace('    preload_start();\n    ram_pin_start();\n', '    ram_pin_start();   // before the preload: it then reads the pinned copies\n    preload_start();\n')
    else:
        sys.exit('FAILED: "preload_start(); ram_pin_start();" sequence not found in model.cpp')
    wr(p, s)
print('ok: page-locked copy applied')
