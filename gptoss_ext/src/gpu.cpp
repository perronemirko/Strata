// CPU-emulated device: the "VRAM" is malloc, every operation executes immediately with the host kernels. Same API as gpu_cuda.cu.
#include "gptoss/gpu.hpp"
#include "gptoss/ops.hpp"
#include "gptoss/weights.hpp"

#include <algorithm>
#include <cmath>
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
void bind_thread() {}
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

// Device-resident activations: the emulated device runs the SAME math as the host path (ops.cpp / Model::attn_heads),
// so it is the executable specification of the CUDA kernels in gpu_cuda.cu and the tests compare against the CPU path.
bool d2h(void* h, const void* d, size_t n) { std::memcpy(h, d, n); return true; }
bool dmm_bf16(const void* dW, int rows, int cols, const float* dx, int N, float* dy) { return matmat_bf16(dW, rows, cols, dx, N, dy); }
bool rms_rows(const float* dx, const float* dg, float eps, int N, int H, float* dy) {
    for (int n = 0; n < N; ++n) gptoss::rmsnorm(dx + (size_t) n * H, dg, eps, H, dy + (size_t) n * H);
    return true;
}
bool add_rows(float* dX, const float* dA, const float* db, int N, int H) {
    for (int n = 0; n < N; ++n) for (int i = 0; i < H; ++i) dX[(size_t) n * H + i] += (dA ? dA[(size_t) n * H + i] : 0.0f) + (db ? db[i] : 0.0f);
    return true;
}
bool qk_rope_kv(float* dq, float* dk, const float* dv, const float* bq, const float* bk, const float* bv, const float* cs, const float* sn,
                int N, int p0, const AttnShape& a, float* Kc, float* Vc) {
    const int D = a.D, half = D / 2, QD = a.heads * D, KD = a.kv_heads * D;
    for (int t = 0; t < N; ++t) {
        const int pos = p0 + t;
        const size_t slot = a.sliding ? (size_t) pos % (size_t) a.cap : (size_t) pos;
        float* q = dq + (size_t) t * QD; float* k = dk + (size_t) t * KD;
        for (int i = 0; i < QD; ++i) q[i] += bq[i];
        for (int i = 0; i < KD; ++i) k[i] += bk[i];
        for (int h = 0; h < a.heads; ++h) rotary(q + (size_t) h * D, D, cs + (size_t) pos * half, sn + (size_t) pos * half, a.rope_pairs);
        for (int h = 0; h < a.kv_heads; ++h) rotary(k + (size_t) h * D, D, cs + (size_t) pos * half, sn + (size_t) pos * half, a.rope_pairs);
        std::memcpy(Kc + slot * KD, k, sizeof(float) * (size_t) KD);
        for (int i = 0; i < KD; ++i) Vc[slot * KD + (size_t) i] = dv[(size_t) t * KD + i] + bv[i];
    }
    return true;
}
bool attn_supported(const AttnShape&) { return true; }   // the emulated device runs the host math: any shape is the spec
bool attn_rows(const float* dq, const float* Kc, const float* Vc, const float* sinks, int N, int p0, const AttnShape& a, float* out) {
    const int D = a.D, QD = a.heads * D, KD = a.kv_heads * D, rep = a.heads / a.kv_heads;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> sc;
    for (int t = 0; t < N; ++t) {
        const int pos = p0 + t, lo = a.sliding ? std::max(0, pos - a.window + 1) : 0;
        sc.assign((size_t) (pos - lo + 1), 0.0f);
        for (int hd = 0; hd < a.heads; ++hd) {
            const float* q = dq + (size_t) t * QD + (size_t) hd * D;
            const int g = hd / rep;
            float mx = sinks[hd];
            for (int u = lo; u <= pos; ++u) {
                const size_t slot = a.sliding ? (size_t) u % (size_t) a.cap : (size_t) u;
                const float* k = Kc + slot * KD + (size_t) g * D;
                float s = 0; for (int j = 0; j < D; ++j) s += q[j] * k[j];
                sc[(size_t) (u - lo)] = s * scale; mx = std::max(mx, sc[(size_t) (u - lo)]);
            }
            double z = std::exp((double) sinks[hd] - mx);
            for (int u = lo; u <= pos; ++u) { sc[(size_t) (u - lo)] = std::exp(sc[(size_t) (u - lo)] - mx); z += sc[(size_t) (u - lo)]; }
            float* o = out + (size_t) t * QD + (size_t) hd * D;
            for (int j = 0; j < D; ++j) o[j] = 0;
            for (int u = lo; u <= pos; ++u) {
                const size_t slot = a.sliding ? (size_t) u % (size_t) a.cap : (size_t) u;
                const float p = (float) (sc[(size_t) (u - lo)] / z);
                const float* v = Vc + slot * KD + (size_t) g * D;
                for (int j = 0; j < D; ++j) o[j] += p * v[j];
            }
        }
    }
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
