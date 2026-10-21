// gptoss prefill_dev - STUB: no device-resident prefill. The CPU build links this so src/model.cpp's pdev:: calls
// resolve; the Model never reaches them, because available() is false and gpu_setup leaves pdev_on false.
#include "gptoss/prefill_dev.hpp"

namespace gptoss {
namespace pdev {

bool available() { return false; }

bool init(const Cfg&, const char*, size_t) { return false; }
int chosen_chunk() { return 0; }
size_t reserved_bytes() { return 0; }
void shutdown() {}
bool upload_layer(int, bool, const float*, const float*, const float*, const float*, const float*, const float*,
                  const float*, const float*, const void*, const void*, const void*, const void*, const void*) { return false; }
void set_rope(const float*, const float*, int) {}
bool x_upload(const float*, int) { return false; }
bool x_download(float*, int) { return false; }
bool kv_prefetch(int, const float*, const float*, int, bool, int) { return false; }
bool run_layer(const Cfg&, int, const LayerIO&, int, int, bool, float*) { return false; }
void kv_store_back(int, float*, float*, int, bool, int, int) {}
void kv_trim(int) {}
bool run_experts(const Cfg&, int, const ExpertPtrs*, const int*, const int*, const float*, const int32_t*) { return false; }
void advance(int) {}
bool sync_all() { return false; }
std::string last_error() { return ""; }

}  // namespace pdev
}  // namespace gptoss
