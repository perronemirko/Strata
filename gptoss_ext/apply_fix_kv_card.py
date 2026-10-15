#!/usr/bin/env python3
# Run from the folder that contains gptoss_ext/ (or from inside gptoss_ext/):  python3 apply_fix_kv_card.py
import os, sys
root = 'gptoss_ext' if os.path.isdir('gptoss_ext/src') else '.'
def edit(p, pairs):
    path = os.path.join(root, p)
    s = open(path).read()
    for done_marker, a, b in pairs:
        if done_marker in s:
            continue                                   # already applied
        if s.count(a) != 1:
            sys.exit("FAILED in %s: expected 1 match, found %d of: %r" % (p, s.count(a), a[:80]))
        s = s.replace(a, b)
    open(path, 'w').write(s)

edit('src/prefill_dev.cu', [
 ('int cbase = 0;',
  "    int cov_lo = 0, cov_hi = 0;     // positions already mirrored (banded), to skip needless uploads\n",
  "    int cov_lo = 0, cov_hi = 0;     // positions already mirrored (banded), to skip needless uploads\n    int cbase = 0;                  // banded: the `base` the rows now on the card were written with (a row sits at p - cbase)\n"),
 ('if (base != kv.cbase)',
  "        kv.base = base;\n        const int n = pos0 - base;\n",
  "        kv.base = base;\n        // The card rows of a banded layer sit at (p - base). When the window moved (base changed since the rows were written)\n"
  "        // every mirrored row is at the wrong index: forget the coverage so the window is rebuilt from the host cache below.\n"
  "        if (base != kv.cbase) { kv.cbase = base; kv.cov_lo = kv.cov_hi = base; }\n        const int n = pos0 - base;\n"),
 ('if (kv.valid > pos0) kv.valid = pos0;',
  "    if ((long long) pos0 > kv.valid) {\n",
  "    if (kv.valid > pos0) kv.valid = pos0;   // a rewind: rows from pos0 on are about to be rewritten, none of them is trusted any more\n    if ((long long) pos0 > kv.valid) {\n"),
 ('void kv_trim(int pos) {',
  "bool run_layer(const Cfg& cfg, int layer, const LayerIO& w, int T, int pos0, bool sliding, float* router_logits) {\n",
  "// The host cache was written by the decode (or replaced by a checkpoint) from position `pos` on: whatever the card mirrors\n"
  "// at or past it is stale. Dense layers lower their watermark; banded layers shorten the covered range, so the next\n"
  "// prefetch rebuilds the window from the host cache.\n"
  "void kv_trim(int pos) {\n    if (pos < 0) pos = 0;\n    for (KVDev& kv : g_kv) {\n        if (kv.valid > pos) kv.valid = pos;\n"
  "        if (kv.cov_hi > pos) kv.cov_hi = pos;\n        if (kv.cov_lo > kv.cov_hi) kv.cov_lo = kv.cov_hi;\n    }\n}\n\n"
  "bool run_layer(const Cfg& cfg, int layer, const LayerIO& w, int T, int pos0, bool sliding, float* router_logits) {\n"),
])
edit('include/gptoss/prefill_dev.hpp', [
 ('void kv_trim(int pos);',
  "bool kv_prefetch(int layer, const float* host_K, const float* host_V, int cap, bool sliding, int pos0);\n",
  "bool kv_prefetch(int layer, const float* host_K, const float* host_V, int cap, bool sliding, int pos0);\nvoid kv_trim(int pos);                              // the host KV changed from `pos` on (decode, checkpoint): the card's mirror is stale from there\n"),
])
edit('src/prefill_dev_stub.cpp', [
 ('void kv_trim(int)',
  "void kv_store_back(int, float*, float*, int, bool, int, int) {}\n",
  "void kv_store_back(int, float*, float*, int, bool, int, int) {}\nvoid kv_trim(int) {}\n"),
])
edit('src/model.cpp', [
 ('int pdev_kv_pos = 0;',
  "    int dev_kv_pos = 0; ",
  "    int pdev_kv_pos = 0;                               // the same watermark for the pdev (prefill_dev.cu) mirror, which is separate from attn_dev's\n    int dev_kv_pos = 0; "),
 ('p_->pdev_kv_pos = 0; }',
  "void Model::reset() { p_->cached.clear(); p_->dev_kv_pos = 0; }",
  "void Model::reset() { p_->cached.clear(); p_->dev_kv_pos = 0; p_->pdev_kv_pos = 0; }"),
 ('s.pdev_kv_pos = 0;\n',
  "    s.dev_kv_pos = 0;                                  // the host KV was replaced",
  "    s.pdev_kv_pos = 0;\n    s.dev_kv_pos = 0;                                  // the host KV was replaced"),
 ('if (s.pdev_kv_pos > pos)',
  "    if (s.dev_kv_pos > pos) s.dev_kv_pos = pos;",
  "    if (s.pdev_kv_pos > pos) s.pdev_kv_pos = pos;\n    if (s.dev_kv_pos > pos) s.dev_kv_pos = pos;"),
 ('pdev::kv_trim(pdev_kv_pos);',
  "    preload_wait();   // the background preload holds every slot busy",
  "    pdev::kv_trim(pdev_kv_pos);   // what the decode / a checkpoint wrote into the host KV since the last device prefill is not on the card\n    preload_wait();   // the background preload holds every slot busy"),
 ('pdev_kv_pos = p0 + T;',
  "        guard.ok = true;\n",
  "        guard.ok = true;\n        pdev_kv_pos = p0 + T;\n"),
])
print('ok: card KV fixes applied')
