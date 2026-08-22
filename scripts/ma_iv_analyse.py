"""Score the IV decomposition: does tb_both's advantage concentrate where an instrument exists?

Reads `results/iv_decomposition.json` (written by `scripts/ma_iv_decomposition.py`) and runs
the difference-in-differences that the IV account predicts:

    advantage(episode)  = success(tb_both) - success(tb_clamp)        same graph, both policies
    DiD                 = mean advantage | IV present
                        - mean advantage | IV absent

    IV account predicts DiD > 0. A DiD indistinguishable from zero says the +0.021 is
    something OTHER than instrument value, which is equally worth knowing.

Bootstrapped over SEEDS, not episodes. Episodes within a seed share a trained policy pair
and are not independent; resampling them would understate the interval, which is the error
that produced a retracted figure earlier in this project.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def per_seed_rows(blob) -> list:
    rows = []
    for entry in blob:
        both, clamp = entry["both"], entry["clamp"]
        adv_iv, adv_no = [], []
        for b, c in zip(both, clamp):
            adv = float(b["success"]) - float(c["success"])
            (adv_iv if b["iv_structure"] else adv_no).append(adv)
        rows.append({
            "seed": entry["seed"],
            "n_iv": len(adv_iv), "n_no": len(adv_no),
            "adv_iv": float(np.mean(adv_iv)) if adv_iv else np.nan,
            "adv_no": float(np.mean(adv_no)) if adv_no else np.nan,
            "adv_all": float(np.mean(adv_iv + adv_no)),
            "vary_rate": float(np.mean([r["varies"] / max(1, r["varies"] + r["clamps"])
                                        for r in both])),
        })
    return rows


def boot_ci(values, n=20000, seed=0):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean()
                      for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="results/iv_decomposition.json")
    args = ap.parse_args(argv)

    blob = json.loads(pathlib.Path(args.path).read_text())
    rows = per_seed_rows(blob)

    print("seed   n_IV  n_noIV   adv|IV   adv|noIV      DiD   vary%")
    for r in rows:
        did = r["adv_iv"] - r["adv_no"]
        print("  %2d    %4d   %5d   %+.4f    %+.4f   %+.4f   %4.1f%%" % (
            r["seed"], r["n_iv"], r["n_no"], r["adv_iv"], r["adv_no"], did,
            100 * r["vary_rate"]))

    did_values = [r["adv_iv"] - r["adv_no"] for r in rows]
    m_all, lo_all, hi_all = boot_ci([r["adv_all"] for r in rows])
    m_iv, lo_iv, hi_iv = boot_ci([r["adv_iv"] for r in rows])
    m_no, lo_no, hi_no = boot_ci([r["adv_no"] for r in rows])
    m_did, lo_did, hi_did = boot_ci(did_values)

    def line(label, m, lo, hi):
        mark = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print("  %-28s %+.4f   CI [%+.4f, %+.4f]   %s" % (label, m, lo, hi, mark))

    print("\nbootstrapped over %d seeds" % len(rows))
    line("advantage, all episodes", m_all, lo_all, hi_all)
    line("advantage | IV present", m_iv, lo_iv, hi_iv)
    line("advantage | IV absent", m_no, lo_no, hi_no)
    line("DiD (IV - noIV)", m_did, lo_did, hi_did)

    print()
    if np.isfinite(lo_did) and lo_did > 0:
        print("  READING: advantage is larger where an instrument exists. Consistent with")
        print("  vary carrying instrument value, which clamp-only forfeits.")
    elif np.isfinite(hi_did) and hi_did < 0:
        print("  READING: advantage is SMALLER where an instrument exists -- opposite to the")
        print("  IV account. Whatever drives the +0.021, it is not instrument value.")
    else:
        print("  READING: DiD indistinguishable from zero. The IV account is NOT supported")
        print("  by this test; the +0.021 has some other source. Note the test is only as")
        print("  powerful as the seed count allows -- report the interval, not a verdict.")
    return rows


if __name__ == "__main__":
    main()
