// Interfaccia comune di ogni modello. Un adapter per modello implementa questa classe.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gptoss {

struct ModelAdapter {
    virtual ~ModelAdapter() = default;
    virtual const char* arch() const = 0;   // architettura, uguale a "arch" in arch.json (arch-spec/1)
    virtual int vocab() const = 0;
    virtual void reset() = 0;
    // Il prefill riempie la cache per le posizioni [pos0, pos0+n); i logit dell'ultimo token vanno in *logits.
    virtual void prefill(const int* ids, int n, int pos0, std::vector<float>* logits) = 0;
    // Decodifica di un token alla posizione pos.
    virtual void forward(int token, int pos, std::vector<float>* logits) = 0;
};

// Costruisce un adapter dalla cartella del modello. Le opzioni (RunOpts, ecc.) sono catturate dalla factory.
using AdapterFactory = std::function<std::unique_ptr<ModelAdapter>(const std::string& model_dir)>;

}  // namespace gptoss
