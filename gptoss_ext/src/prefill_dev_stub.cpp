// gptoss/prefill_dev_stub.cpp - emulated/CPU build: no CUDA implementation is linked, so available() is false and Model
// keeps the old prefill. The symbols exist so model.cpp compiles unchanged in both builds.
#include "gptoss/prefill_dev.hpp"

namespace gptoss {
namespace pdev {

bool available() { return false; }

bool init(const Cfg&, const char*, size_t) { return false; }
void shutdown() {}
void set_rope(const float*, const float*, int) {}
bool upload_layer(int, bool, const float*, const float*, const float*, const float*, const float*, const float*,
                  const float*, const float*, const void*, const void*, const void*, const void*, const void*) { return false; }

float* x_dev() { return nullptr; }
bool x_upload(const float*, int) { return false; }
bool x_download(float*, int) { return false; }
void kv_prefetch(int, const float*, const float*, int, bool, int) {}
bool run_layer(const Cfg&, int, const LayerIO&, int, int, bool, float*) { return false; }
bool run_experts(const Cfg&, int, const ExpertPtrs*, const int*, const int*, const float*, const int*) { return false; }
void kv_store_back(int, float*, float*, int, bool, int, int) {}
bool sync_all() { return true; }
void advance(int) {}

}  // namespace pdev
}  // namespace gptoss
