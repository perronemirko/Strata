// Adapter di gpt-oss: avvolge gptoss::Model. Nessun calcolo nuovo: ogni chiamata va dritta al Model gia' verificato col selfcheck.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>

#include "adapter.hpp"
#include "gateway.hpp"
#include "model.hpp"

namespace gptoss {

class GptOssAdapter final : public ModelAdapter {
public:
    explicit GptOssAdapter(const RunOpts& o) : opts_(o) {}

    bool open(const std::string& dir, std::string& err) { return m_.load(dir, opts_, err); }
    Model& model() { return m_; }

    const char* arch() const override { return m_.config().arch.c_str(); }
    int vocab() const override { return m_.config().vocab; }
    void reset() override { m_.reset(); }
    void prefill(const int* ids, int n, int pos0, std::vector<float>* logits) override { m_.prefill(ids, n, pos0, logits); }
    void forward(int token, int pos, std::vector<float>* logits) override { m_.forward(token, pos, logits); }

private:
    RunOpts opts_;
    Model m_;
};

// Factory da registrare nel gateway: apre il modello con le opzioni del run.
inline AdapterFactory gpt_oss_factory(const RunOpts& o) {
    return [o](const std::string& dir) -> std::unique_ptr<ModelAdapter> {
        auto a = std::make_unique<GptOssAdapter>(o);
        std::string err;
        if (!a->open(dir, err)) throw std::runtime_error("gpt_oss: " + err);
        return a;
    };
}

// Registra gpt_oss nel gateway (chiamare una volta all'avvio, con le RunOpts del run).
inline void register_gpt_oss(const RunOpts& o) {
    Gateway::instance().add("gpt_oss", gpt_oss_factory(o));
}

}  // namespace gptoss
