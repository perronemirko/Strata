void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.3125 bpw (de)-quantization

void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.5625 bpw (de)-quantization

void dequantize_row_iq2_s(const block_iq2_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ====================== 3.0625 bpw (de)-quantization

void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

// ====================== 3.3125 bpw (de)-quantization

void dequantize_row_iq3_s(const block_iq3_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// ====================== 1.5625 bpw (de)-quantization

void dequantize_row_iq1_s(const block_iq1_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

void dequantize_row_iq1_m(const block_iq1_m * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float delta[4];
    uint16_t idx[4];

    iq1m_scale_t scale;

    for (int i = 0; i < nb; i++) {

        const uint16_t * sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = GGML_FP16_TO_FP32(scale.f16);

        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}


void dequantize_row_iq4_nl(const block_iq4_nl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

void dequantize_row_iq4_xs(const block_iq4_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}
