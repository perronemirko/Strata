// Test proprietà delle primitive hyper-connection (non c'è un riferimento numerico: si verificano invarianti matematici).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "gptoss/hc_ops.hpp"

int main() {
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    bool ok = true;
    auto check = [&](bool c, const char* what) { std::printf("%s %s\n", c ? "OK  " : "FAIL", what); ok = ok && c; };

    // 1) RMS norm: media dei quadrati ~ 1
    {
        std::vector<float> x(256), y(256);
        for (auto& v : x) v = U(rng) * 3.0f;
        gptoss::rms_norm_plain(x.data(), 256, 1e-9f, y.data());
        double s = 0; for (float v : y) s += (double) v * v;
        check(std::fabs(s / 256 - 1.0) < 1e-4, "rms_norm: media dei quadrati = 1");
    }
    // 2) Sinkhorn: righe e colonne sommano a ~1 (eps piccolo)
    {
        const int64_t hc = 4;
        std::vector<float> C(hc * hc);
        for (auto& v : C) v = U(rng) * 2.0f;
        gptoss::hc_sinkhorn(C.data(), hc, 30, 1e-6f);
        double maxerr = 0;
        for (int64_t s = 0; s < hc; ++s) { double r = 0; for (int64_t d = 0; d < hc; ++d) r += C[s * hc + d]; maxerr = std::fmax(maxerr, std::fabs(r - 1)); }
        for (int64_t d = 0; d < hc; ++d) { double c = 0; for (int64_t s = 0; s < hc; ++s) c += C[s * hc + d]; maxerr = std::fmax(maxerr, std::fabs(c - 1)); }
        std::printf("     errore massimo righe/colonne Sinkhorn: %.2e\n", maxerr);
        check(maxerr < 1e-3, "sinkhorn: matrice doppiamente stocastica");
    }
    // 3) hc_post con post=1 e comb identità: out[dst] = x_new + residuo[dst]
    {
        const int64_t hc = 4, ne = 8;
        std::vector<float> xn(ne), res(hc * ne), out(hc * ne), post(hc, 1.0f), comb(hc * hc, 0.0f);
        for (auto& v : xn) v = U(rng);
        for (auto& v : res) v = U(rng);
        for (int64_t h = 0; h < hc; ++h) comb[h * hc + h] = 1.0f;
        // post=1 e comb identità: out[dst] = xn + res[dst], ma xn è aggiunto una volta per dst
        gptoss::hc_post(xn.data(), res.data(), post.data(), comb.data(), hc, ne, out.data());
        double maxerr = 0;
        for (int64_t d = 0; d < hc; ++d) for (int64_t e = 0; e < ne; ++e)
            maxerr = std::fmax(maxerr, std::fabs(out[d * ne + e] - (xn[e] + res[d * ne + e])));
        check(maxerr < 1e-6, "hc_post con comb identità = x_new + residuo");
    }
    // 4) hc_pre + hc_head: forme e finitezza su input casuali
    {
        const int64_t hc = 4, ne = 16, hc_dim = hc * ne, mix = (2 + hc) * hc;
        std::vector<float> x(hc_dim), fn(mix * hc_dim), scale = {0.5f, 0.5f, 0.5f}, base(mix, 0.0f), ox(ne), opost(hc), ocomb(hc * hc);
        for (auto& v : x) v = U(rng);
        for (auto& v : fn) v = U(rng) * 0.1f;
        gptoss::hc_pre(x.data(), hc, ne, fn.data(), scale.data(), base.data(), 1e-5f, 1e-6f, 20, ox.data(), opost.data(), ocomb.data());
        bool fin = true;
        for (float v : ox) fin = fin && std::isfinite(v);
        for (float v : opost) fin = fin && std::isfinite(v) && v > 0 && v < 2;
        check(fin, "hc_pre: valori finiti, post in (0,2)");
        std::vector<float> hx(ne), hfn(hc * hc_dim), hscale = {0.5f}, hbase(hc, 0.0f);
        for (auto& v : hfn) v = U(rng) * 0.1f;
        gptoss::hc_head(x.data(), hc, ne, hfn.data(), hscale.data(), hbase.data(), 1e-5f, 1e-6f, hx.data());
        bool fin2 = true; for (float v : hx) fin2 = fin2 && std::isfinite(v);
        check(fin2, "hc_head: valori finiti");
    }
    std::printf(ok ? "TUTTI MATCH\n" : "ERRORI\n");
    return ok ? 0 : 1;
}
