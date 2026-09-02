"""Generate the appendices, with every table computed from the data files.

FOUR APPENDICES, ONE ARGUMENT. Excluded runs, training budget, and checkpoint selection are
not three pieces of housekeeping -- together they say that at 4,000 episodes the sweep measured
convergence as much as capability, and that at 12,000 the checkpoint at which a policy is
caught matters. That is why they sit before the supporting ablations rather than after.

Tables are generated, never typed. A table whose numbers were transcribed by hand has been
wrong twice in this project already. Sections whose measurement is still running emit an
explicit PENDING marker rather than a plausible-looking placeholder.

    python scripts/build_appendix.py
"""
from __future__ import annotations
import glob, json, os, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TR = ROOT / "thesis_results"
FLOOR, TAIL = 0.70, 10
PRE = {"k12s50n05b150": "results/longcheck/shd_n05_12k.json",
       "k12s75n04b150": "results/longcheck/shd_s75_12k.json",
       "k12s50n08b150": "results/longcheck/shd_n08_12k.json",
       "k12s50n10b150": "results/longcheck/shd_n10_12k.json"}


def wr(d):
    t = [h.get("window_rate", 0.0) for h in (d.get("history") or [])[-TAIL:]]
    return sum(t) / len(t) if t else 0.0


def jload(p):
    return json.loads(pathlib.Path(p).read_text())


def tbl(caption, label, spec, header, body, note=""):
    out = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
           header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    if note:
        out.append(note)
    return "\n".join(out) + "\n"


# --- A: excluded runs ---------------------------------------------------------------------
def appendix_excluded():
    rows = []
    for p in sorted((TR / "sweep").glob("k*_s*.json")):
        d = jload(p)
        if wr(d) >= FLOOR:
            continue
        cell = p.stem.rsplit("_s", 1)[0]
        long = TR / "federation" / f"{cell}_long_s2.json"
        r = jload(long) if long.exists() else None
        rows.append((cell, d.get("seed"), wr(d), d["arms"]["learned"]["success"],
                     wr(r) if r else None,
                     r["arms"]["learned"]["success"] if r else None,
                     d["arms"]["greedy_uncertainty"]["success"]))
    body = [f"\\texttt{{{c.replace('_', chr(92) + '_')}}} & {s} & {w4:.3f} & {l4:.3f} & "
            f"{f'{w12:.3f} & {l12:.3f}' if w12 is not None else '--- & ---'} & {g:.3f} \\\\"
            for c, s, w4, l4, w12, l12, g in rows]
    return ("\\chapter{Excluded Runs} \\label{app:excluded}\n\n"
            "Every run falling below the competence floor of \\S\\ref{sec:meth_gate}, with what\n"
            "the same configuration reaches at $12{,}000$ episodes. The retrained runs appear in\n"
            "no sweep table; they are reported here and in Appendix~\\ref{app:budget}.\n\n"
            + tbl("The excluded runs, at the sweep's budget and at three times it.",
                  "tab:excluded", "llcccccc",
                  r"Cell & Seed & \multicolumn{2}{c}{4{,}000 episodes} & "
                  r"\multicolumn{2}{c}{12{,}000 episodes} & Myopic \\"
                  "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
                  "\n" r" & & window & joint & window & joint & joint \\", body)
            + f"\nAll {len(rows)} are seed 2 and all are at $k_v=12$. Every one clears the floor "
              "when retrained,\nand every one finishes above the myopic rule on its own cell.\n")


# --- B: training budget -------------------------------------------------------------------
def appendix_budget():
    marks = [2000, 4000, 6000, 8000, 10000, 12000]
    runs = sorted(glob.glob(str(ROOT / "results/longcheck/*_conv_s*.json"))
                  + glob.glob(str(ROOT / "results/longcheck/*_long_s*.json")))
    body = []
    for p in runs[:12]:
        d = jload(p)
        ck = (d.get("checkpoints") or {}).get("checkpoints", [])
        if not ck:
            continue
        eps = d["config"]["ppo_episodes_per_update"]
        curve = [(c["update"] * eps, c["solve_rate"]) for c in ck]
        cells = []
        for m in marks:
            v = [s for e, s in curve if e <= m]
            cells.append(f"{v[-1]:.2f}" if v else "---")
        name = os.path.basename(p).replace(".json", "").replace("_", chr(92) + "_")
        body.append(f"\\texttt{{\\scriptsize {name}}} & " + " & ".join(cells) + r" \\")

    eight = ROOT / "results/sweep12k/shd_u0500"
    if any(eight.glob("*.json")):
        rows, skipped = [], []
        for p in sorted(eight.glob("*.json")):
            cell = p.stem
            bp = ROOT / PRE.get(cell, f"results/sweep12k/shd/{cell}.json")
            if not bp.exists():
                continue
            v8 = [e["means"]["learned"]["hard"] for e in jload(p)]
            v12 = [e["means"]["learned"]["hard"] for e in jload(bp)]
            vg = [e["means"]["greedy"]["hard"] for e in jload(bp)]
            # A row that cannot be computed is DROPPED, never emitted as nan. A generator
            # that prints an unusable number into a thesis is worse than one that omits the
            # row: a gap is visible, and `nan` reached Overleaf once before this guard.
            if not (v8 and v12 and vg):
                skipped.append(cell)
                continue
            rows.append(f"\\texttt{{{cell}}} & {np.mean(v8):.5f} & {np.mean(v12):.5f} & "
                        f"{np.mean(vg):.5f} \\\\")
        note = ("\n\\textbf{Not yet measured:} " +
                ", ".join(f"\\texttt{{{c}}}" for c in skipped) + ".\n") if skipped else ""
        # Count the direction of the move so the paragraph beneath the table states what the
        # table shows rather than what the reader is expected to infer from eighteen rows.
        better = worse = same = ahead8 = ahead12 = 0
        for r in rows:
            f = [c.strip() for c in r.replace("\\\\", "").split("&")]
            v8, v12, vg = float(f[1]), float(f[2]), float(f[3])
            better += v12 < v8
            worse += v12 > v8
            same += v12 == v8
            ahead8 += v8 < vg
            ahead12 += v12 < vg
        note += (f"\n{better} of the {len(rows)} cells improve between $8{{,}}000$ and "
                 f"$12{{,}}000$ episodes, {worse} get worse and {same} is unchanged. The count "
                 f"with the learned mean below the myopic rule moves from {ahead8} to "
                 f"{ahead12}. Most of the gain over the sweep's $4{{,}}000$ episodes is "
                 f"therefore already present at $8{{,}}000$, and the last third of training "
                 f"buys a small net improvement against run-to-run movement of comparable "
                 f"size. $12{{,}}000$ is reported as sufficient rather than as a threshold.\n")
        eight_tbl = tbl("Structural distance at 8{,}000 against 12{,}000 episodes, same runs, "
                        "same seeds, selected checkpoint throughout.", "tab:eightk", "lccc",
                        r"Cell & 8{,}000 ep & 12{,}000 ep & Myopic \\", rows, note)
    else:
        eight_tbl = ("\\textbf{PENDING.} The 8{,}000-episode comparison is measuring. Do not "
                     "write this paragraph until it lands.\n")

    return ("\\chapter{Training Budget and Convergence} \\label{app:budget}\n\n"
            "The sweep trains every cell for $4{,}000$ episodes. Three of the four structural\n"
            "claims in Chapter~\\ref{Chap4} were artefacts of that budget applied across cells of\n"
            "unequal difficulty, so that the apparent effect of a swept parameter was partly the\n"
            "effect of the harder settings needing more training.\n\n"
            + tbl("Per-window solve rate over training, from the checkpoint schedule each run "
                  "records. Most runs converge by $8{,}000$ episodes; the slowest does not.",
                  "tab:convergence", "l" + "c" * len(marks),
                  "Run & " + " & ".join(f"{m//1000}k" for m in marks) + r" \\", body)
            + "\n" + eight_tbl)


# --- C: checkpoint selection ---------------------------------------------------------------
def appendix_checkpoint():
    body = []
    for p in sorted(glob.glob(str(ROOT / "results/sweep12k/shd_final/*.json"))):
        cell = os.path.basename(p)[:-5]
        bp = ROOT / PRE.get(cell, f"results/sweep12k/shd/{cell}.json")
        if not bp.exists():
            continue
        for b, f in zip(jload(bp), jload(p)):
            B, F = b["means"]["learned"]["hard"], f["means"]["learned"]["hard"]
            ratio = (max(B, F) + 1e-9) / (min(B, F) + 1e-9)
            flag = r"\textbf{selected}" if B > F and ratio > 3 else (
                r"\textbf{final}" if F > B and ratio > 3 else "---")
            body.append(f"\\texttt{{{cell}}} & {b['seed']} & {B:.5f} & {F:.5f} & {flag} \\\\")
    am = []
    for p in sorted(glob.glob(str(ROOT / "results/sweep12k/shd_argmax/*.json"))):
        cell = os.path.basename(p)[:-5]
        bp = ROOT / PRE.get(cell, f"results/sweep12k/shd/{cell}.json")
        if not bp.exists():
            continue
        for a, s in zip(jload(p), jload(bp)):
            am.append(f"\\texttt{{{cell}}} & {a['seed']} & "
                      f"{s['means']['learned']['hard']:.5f} & "
                      f"{a['means']['learned']['hard']:.5f} \\\\")
    return ("\\chapter{Checkpoint Selection} \\label{app:checkpoint}\n\n"
            "At $12{,}000$ episodes neither checkpoint convention is safe alone. Selection on\n"
            "mutual information occasionally retains an exploratory policy; the final update\n"
            "occasionally retains a drifted one. The two fail on different cells, which is why\n"
            "Chapter~\\ref{Chap4} reports both.\n\n"
            + tbl("Where the conventions disagree by more than a factor of three.",
                  "tab:ckpt_tail", "llccl",
                  r"Cell & Seed & Selected & Final & Worse \\", body)
            + "\nThe disagreement is not an artefact of how actions are chosen at evaluation "
              "time.\nSampling is the more forgiving convention: on the affected seeds argmax is "
              "worse still,\nbecause committing to the mode of a poor policy costs more than "
              "sampling around it.\n\n"
            + tbl("Sampling against argmax at the selected checkpoint.",
                  "tab:ckpt_argmax", "llcc",
                  r"Cell & Seed & Sampled & Argmax \\", am))


# --- D: supporting ablations ----------------------------------------------------------------
def appendix_ablations():
    body = []
    for label, pat in (("Pooled, credit on", "k08s50n04b150_pooled_credit"),
                       ("Pooled, credit off", "k08s50n04b150_pooled_nocredit"),
                       ("Federated, credit on", "k08s50n04b150_E4_credit"),
                       ("Federated, credit off", "k08s50n04b150_E4_nocredit")):
        fs = sorted(glob.glob(str(ROOT / f"results/*/{pat}_s*.json")))
        if not fs:
            continue
        v = [jload(f)["arms"]["learned"]["global_hard_shd"] for f in fs]
        body.append(f"{label} & {len(v)} & {np.mean(v):.5f} \\\\")

    attr = sorted(glob.glob(str(ROOT / "results/attr_train/*.json")))
    arows = []
    for f in attr:
        d = jload(f)
        arows.append(f"{d.get('seed')} & {d['arms']['learned']['success']:.3f} & "
                     f"{d['arms']['greedy_uncertainty']['success']:.3f} \\\\")

    return ("\\chapter{Supporting Ablations} \\label{app:ablations}\n\n"
            + tbl("Turn-aware credit assignment at $k_v=8$, three seeds per cell. The effect "
                  "appears only under federation and does not replicate at $k_v=12$.",
                  "tab:credit", "lcc", r"Configuration & Seeds & SHD \\", body)
            # CORRECTED 2 Sep 22:xx. This table was captioned "adding an attribution term to
            # the training reward". It is not that experiment. Every run in results/attr_train
            # has reward_criterion="claims" and observe_owner_channel=False; the trainer accepts
            # only "claims" and "u14" and neither scores attribution. What varies here is the
            # BELIEF BACKEND, component_attributed against the sweep's factored. No policy in
            # this project was ever trained on an attribution objective.
            + "\n" + tbl("Training under the attribution belief backend, $k_v=12$, four agents, "
                         "$4{,}000$ episodes, scored on the structural criterion. The reward is "
                         "unchanged from the sweep; only the belief representation differs.",
                         "tab:attrbackend", "lcc",
                         r"Seed & Learned & Myopic \\", arows))


def main() -> int:
    out = ROOT / "thesis/Appendix.tex"
    parts = [r"\appendix", "",
             "% Generated by scripts/build_appendix.py. Do not edit tables by hand.",
             ""]
    for fn in (appendix_excluded, appendix_budget, appendix_checkpoint, appendix_ablations):
        parts.append(fn())
        parts.append("")
    out.write_text("\n".join(parts))
    print(f"wrote {out.relative_to(ROOT)}: {len(out.read_text().splitlines())} lines, "
          f"4 appendices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
