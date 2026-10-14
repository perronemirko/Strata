// CPU-emulated device: the "VRAM" is malloc, every operation executes immediately with the host kernels. Same API as gpu_cuda.cu.
#include "gptoss/gpu.hpp"
#include "gptoss/ops.hpp"
#include "gptoss/weights.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace gptoss {
namespace gpu {

namespace {
struct EmuCtx { std::vector<float> y; int N = 0, H = 0; };
EmuCtx g_emu[5];
struct EmuDenseCtx {
    std::vector<float> y;
    int n = 0, count = 0;
    std::vector<size_t> off;
    std::vector<int> rows;
};
EmuDenseCtx g_edc[kDenseCtxBatch];
int g_events = 0;
}  // namespace

bool is_emulated() { return true; }
bool init(int, std::string&) { return true; }
const char* device_name() { return "CPU-emulated device"; }
void mem_info(size_t* f, size_t* t) { *t = (size_t) 16 << 30; *f = *t; }
void* alloc(size_t n) { return std::malloc(n ? n : 1); }
void release(void* p) { std::free(p); }
bool h2d(void* d, const void* h, size_t n) { std::memcpy(d, h, n); return true; }
void set_options(bool) {}
bool reserve_scratch(size_t, size_t, int, int, int, size_t) { return true; }
std::string last_error() { return ""; }
bool last_oom() { return false; }

bool matmat_bf16(const void* dW, int rows, int cols, const float* hx, int N, float* hy) {
    WView w; w.p = (const uint8_t*) dW; w.rows = rows; w.cols = cols; w.dt = DType::BF16;
    gptoss::matmat(w, hx, N, hy);
    return true;
}

bool matmat_mx(const void* dB, const void* dS, int rows, int cols, const float* hx, int N, float* hy) {
    MxView w; w.blocks = (const uint8_t*) dB; w.scales = (const uint8_t*) dS; w.rows = rows; w.cols = cols;
    for (int n = 0; n < N; ++n) gptoss::matvec_mx(w, hx + (size_t) n * cols, hy + (size_t) n * rows);
    return true;
}

bool expert_run_async(int ci, const ExpertDesc& d, const float* hx, const float* wt, int N) {
    if (ci < 0 || ci > kExpertCtxBatch) return false;
    const int H = d.H, I = d.I;
    EmuCtx& c = g_emu[ci];
    c.N = N; c.H = H;
    c.y.assign((size_t) N * (size_t) H, 0.0f);
    MxView gu; gu.blocks = (const uint8_t*) d.gu_b; gu.scales = (const uint8_t*) d.gu_s; gu.rows = 2 * I; gu.cols = H;
    MxView dn; dn.blocks = (const uint8_t*) d.dn_b; dn.scales = (const uint8_t*) d.dn_s; dn.rows = H; dn.cols = I;
    const uint16_t* gb = (const uint16_t*) d.gu_bias;
    const uint16_t* db = (const uint16_t*) d.dn_bias;
    std::vector<float> g((size_t) 2 * I), act((size_t) I), eb((size_t) H);
    for (int n = 0; n < N; ++n) {
        gptoss::matvec_mx(gu, hx + (size_t) n * H, g.data());
        for (int i = 0; i < 2 * I; ++i) g[(size_t) i] += bf16_to_f32(gb[i]);
        for (int j = 0; j < I; ++j) {
            const float gate = d.halves ? g[(size_t) j] : g[(size_t) 2 * j];
            const float up = d.halves ? g[(size_t) I + j] : g[(size_t) 2 * j + 1];
            act[(size_t) j] = swiglu(gate, up, d.limit, d.alpha);
        }
        gptoss::matvec_mx(dn, act.data(), eb.data());
        for (int j = 0; j < H; ++j) c.y[(size_t) n * H + j] = wt[n] * (eb[(size_t) j] + bf16_to_f32(db[j]));
    }
    return true;
}

bool expert_collect(int ci, float* hy) {
    if (ci < 0 || ci > kExpertCtxBatch) return false;
    const EmuCtx& c = g_emu[ci];
    std::memcpy(hy, c.y.data(), c.y.size() * sizeof(float));
    return true;
}

// The emulated device has no streams: the work executes at once, and dense_collect just hands it back. Same
// results as the host path, so the tests that compare against the CPU path still pass.
bool dense_run_async(int ci, const float* hx, int N, const DenseOp* ops, int count) {
    if (ci < 0 || ci >= kDenseCtxBatch || count <= 0 || count > kDenseMaxOps || N <= 0) return false;
    for (int j = 0; j < count; ++j) if (!ops[j].W || ops[j].rows <= 0 || ops[j].cols <= 0) return false;
    EmuDenseCtx& c = g_edc[ci];
    c.n = N; c.count = count;
    c.off.assign((size_t) count + 1, 0);
    c.rows.assign((size_t) count, 0);
    size_t total = 0;
    for (int j = 0; j < count; ++j) { c.rows[j] = ops[j].rows; c.off[(size_t) j] = total; total += (size_t) N * (size_t) ops[j].rows; }
    c.off[(size_t) count] = total;
    c.y.assign(total, 0.0f);
    for (int j = 0; j < count; ++j) {
        WView w; w.p = (const uint8_t*) ops[j].W; w.rows = ops[j].rows; w.cols = ops[j].cols; w.dt = DType::BF16;
        gptoss::matmat(w, hx, N, c.y.data() + c.off[(size_t) j]);
    }
    return true;
}

bool dense_collect(int ci, float* const* out) {
    if (ci < 0 || ci >= kDenseCtxBatch) return false;
    const EmuDenseCtx& c = g_edc[ci];
    for (int j = 0; j < c.count; ++j) std::memcpy(out[j], c.y.data() + c.off[(size_t) j], (size_t) c.n * (size_t) c.rows[j] * sizeof(float));
    return true;
}

int event_create() { return g_events++; }
bool event_sync(int) { return true; }
bool h2d_copy_async(void* d, const void* h, size_t n) { std::memcpy(d, h, n); return true; }
bool event_record_copy(int) { return true; }
bool host_register(const void*, size_t) { return true; }

}  // namespace gpu
}  // namespace gptoss
