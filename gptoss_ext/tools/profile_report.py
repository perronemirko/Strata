#!/usr/bin/env python3
"""How many HITS would pinning give? Reads the usage profile and prints, for a number of VRAM slots, the share of routed activations that
land on the N most-used experts (= the hit rate of an ideal static pinning of those N), against the uniform baseline N/(layers*experts).

  python3 tools/profile_report.py /var/tmp/gptoss.prof [--vram-gib 12] [--expert-mib 13.2] [--pin-pct 80]

If the curve is flat (coverage ~ uniform), the profile does not help and the traffic is spread over all experts: the bus will bound the speed."""
import argparse
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--vram-gib", type=float, default=None, help="GiB of VRAM left for experts (after dense weights and KV)")
    ap.add_argument("--expert-mib", type=float, default=13.2)
    ap.add_argument("--pin-pct", type=float, default=80)
    a = ap.parse_args()
    raw = open(a.profile, "rb").read()
    assert raw[:8] == b"GPPRF001", "not a profile"
    L, E = struct.unpack("<II", raw[8:16])
    cnt = list(struct.unpack("<%dQ" % (L * E), raw[16:16 + 8 * L * E]))
    tot = sum(cnt)
    if not tot:
        raise SystemExit("empty profile: run tools/calibrate_profile.py first")
    order = sorted(cnt, reverse=True)
    print("layers %d, experts %d, routed activations %d (= %.0f tokens x %d layers x top-k)" % (L, E, tot, tot / (L * 4), L))
    print("experts never used: %d of %d" % (sum(1 for v in cnt if v == 0), L * E))
    print("\n  slots   %% of experts   hit rate pinned   uniform baseline")
    sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072]
    if a.vram_gib:
        sizes.append(int(a.vram_gib * 1024 / a.expert_mib))
    for n in sorted({min(x, L * E) for x in sizes}):
        cov = 100.0 * sum(order[:n]) / tot
        print("  %5d   %10.1f%%   %14.1f%%   %14.1f%%%s" % (n, 100.0 * n / (L * E), cov, 100.0 * n / (L * E),
              "   <- your VRAM" if a.vram_gib and n == min(int(a.vram_gib * 1024 / a.expert_mib), L * E) else ""))
    if a.vram_gib:
        slots = min(int(a.vram_gib * 1024 / a.expert_mib), L * E)
        npin = int(slots * a.pin_pct / 100)
        print("\nwith %d slots and --pin-pct %g: %d pinned would cover %.1f%% of the activations; the other %d slots are LRU for the rest"
              % (slots, a.pin_pct, npin, 100.0 * sum(order[:npin]) / tot, slots - npin))
    worst = min((100.0 * sum(sorted(cnt[l * E:(l + 1) * E], reverse=True)[:E // 4]) / max(1, sum(cnt[l * E:(l + 1) * E])), l) for l in range(L))
    print("\nlayer with the flattest routing: %d (its top quarter of experts takes %.1f%% of its traffic; uniform = 25%%)" % (worst[1], worst[0]))


if __name__ == "__main__":
    main()
