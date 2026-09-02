"""Generate Chapter 4's skeleton: section scaffold plus every table computed from data.

The prose is deliberately absent. Each section carries a LaTeX comment stating the claim it
must make, the table or figure that carries it, and the boundary where the claim stops
holding -- so the writer is choosing sentences, not numbers.

Every number below is computed from `thesis_results/`, never typed. Re-run after
`scripts/collect_thesis_results.py` and the tables follow the data.
"""
from __future__ import annotations
import glob, json, pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TR = ROOT / "thesis_results"
KS = [4, 8, 12, 20, 30]
CELL = re.compile(r"k(\d+)s(\d+)n(\d+)b(\d+)")
WINDOW_FLOOR = 0.70


def load(folder, pattern="*.json"):
    return [json.loads(p.read_text()) for p in sorted((TR / folder).glob(pattern))]


def sweep_rows():
    rows = []
    for p in sorted((TR / "sweep").glob("*.json")):
        m = CELL.match(p.stem)
        d = json.loads(p.read_text())
        tail = [h.get("window_rate", 0.0) for h in (d.get("history") or [])[-10:]]
        rows.append(dict(k=int(m[1]), sigma=int(m[2]) / 100, n=int(m[3]), beta=int(m[4]) / 100,
                         seed=d.get("seed"), cell=p.stem.rsplit("_s", 1)[0],
                         wr=(sum(tail) / len(tail) if tail else 0.0), arms=d["arms"]))
    return rows


def ckpt(k, which):
    return json.loads((TR / "checkpoint" / f"k{k:02d}_{which}.json").read_text())


def axis_table(rows, key, keep, label, caption, head, xfmt="{:g}"):
    sel = [r for r in rows if keep(r)]
    out = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", r"\begin{tabular}{rccccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{SHD on committed marks} & \multicolumn{3}{c}{Joint recovery rate} & \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           f"{head} & Learned & Myopic & Random & Learned & Myopic & Random & Seeds \\\\", r"\midrule"]
    for x in sorted({r[key] for r in sel}):
        cell = [r for r in sel if r[key] == x and r["wr"] >= WINDOW_FLOOR]
        f = lambda a, m: np.mean([r["arms"][a][m] for r in cell])
        out.append(f"{xfmt.format(x)} & "
                   f"{f('learned','global_hard_shd'):.5f} & {f('greedy_uncertainty','global_hard_shd'):.5f} & "
                   f"{f('random_vary','global_hard_shd'):.5f} & "
                   f"{f('learned','success'):.3f} & {f('greedy_uncertainty','success'):.3f} & "
                   f"{f('random_vary','success'):.3f} & {len(cell)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def checkpoint_table():
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{SHD on committed marks by checkpoint, 200 paired episodes per seed. "
           r"The two agree below the crossover and diverge above it.}",
           r"\label{tab:checkpoint}", r"\begin{tabular}{rccc}", r"\toprule",
           r"$k_v$ & Learned (selected) & Learned (final) & Myopic \\", r"\midrule"]
    for k in KS:
        b = np.mean([r["means"]["learned"]["hard"] for r in ckpt(k, "best")])
        f = np.mean([r["means"]["learned"]["hard"] for r in ckpt(k, "final")])
        g = np.mean([r["means"]["greedy"]["hard"] for r in ckpt(k, "best")])
        out.append(f"{k} & {b:.5f} & {f:.5f} & {g:.5f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def federation_table():
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Joint recovery rate by coordination strategy. Arm E removes the "
           r"information and optimiser partitions; action rights remain partitioned in both.}",
           r"\label{tab:federation}", r"\begin{tabular}{lccccc}", r"\toprule",
           r"Cell & Random & Myopic, fixed partition & Myopic & Learned (federated) & "
           r"Learned (centralised) \\", r"\midrule"]
    for cell, lab in (("k12", "$k_v=12$"), ("k20", "$k_v=20$")):
        A = load("federation", f"v2_{cell}_A_s*.json")
        E = load("federation", f"v2_{cell}_E_s*.json")
        g = lambda runs, key: np.mean([r["arms"][key]["success"] for r in runs
                                       if r["arms"].get(key)])
        out.append(f"{lab} & {g(A,'random_vary'):.3f} & {g(A,'greedy_partitioned'):.3f} & "
                   f"{g(A,'greedy_uncertainty'):.3f} & {g(A,'learned'):.3f} & "
                   f"{g(E,'learned'):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def attribution_tables():
    grid = load("attribution", "attr_ceiling.json") + load("attribution", "attr_ceiling_matched_budget.json")
    flat = [e for d in grid for e in (d if isinstance(d, list) else [d])]
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Latent groups correctly attributed, by peer count. Zero misattributions "
           r"in every cell: the engine identifies the owner or abstains.}",
           r"\label{tab:attribution}", r"\begin{tabular}{rccccc}", r"\toprule",
           r"Peers & Budget & Correct & Incorrect & Total & Share \\", r"\midrule"]
    seen = set()
    for e in sorted(flat, key=lambda e: (e.get("n_agents", 0), e.get("budget", 0))):
        key = (e.get("n_agents"), e.get("budget"), e.get("total"))
        if key in seen or "right" not in e:
            continue
        seen.add(key)
        out.append(f"{e['n_agents'] - 1} & {e['budget']} & {e['right']} & {e.get('wrong', 0)} & "
                   f"{e['total']} & {e['measured']:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    cov = [e for d in load("attribution", "attr_ceiling_budget.json") for e in d]
    out += [r"\begin{table}[htbp]", r"\centering",
            r"\caption{Coverage is a step function. The last three budgets return identical "
            r"counts, not merely equal rates.}", r"\label{tab:coverage}",
            r"\begin{tabular}{rccc}", r"\toprule",
            r"Budget & Correct & Total & Share \\", r"\midrule"]
    for e in cov:
        out.append(f"{e['budget']} & {e['right']} & {e['total']} & {e['measured']:.4f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def note(claim, data, boundary):
    return ("% CLAIM:    " + claim + "\n% DATA:     " + data + "\n% BOUNDARY: " + boundary + "\n")


def main():
    rows = sweep_rows()
    excl = [r for r in rows if r["wr"] < WINDOW_FLOOR]
    parts = [r"\chapter{Results and Analysis} \label{Chap4}", "",
             "% SKELETON ONLY. Every table below is generated by scripts/build_results_skeleton.py",
             "% from thesis_results/ and must not be edited by hand -- edit the data or the script.",
             "% Prose is for the writer. Each section states the claim it must make, the data that",
             "% carries it, and the boundary where the claim stops holding.",
             "% Chapter 3 owns the protocol (sec:meth_eval, sec:meth_paired, sec:meth_baselines,",
             "% sec:meth_gate, sec:meth_ckpt). Do not restate any of it here; cite it.", ""]

    parts += [r"\section{The Sweep} \label{sec:res_sweep}", "",
              note("Learned selection is compared with a myopic uncertainty rule and random "
                   "targeting on four independently swept axes. This section presents the whole "
                   "sweep; the sections after it take one axis each.",
                   f"Figure sweep_grid.pdf; Tables {', '.join(['tab:axis_k','tab:axis_n','tab:axis_sigma','tab:axis_beta'])}. "
                   f"{len(rows)} runs, {len({r['cell'] for r in rows})} cells, final-policy evaluation.",
                   f"{len(excl)} runs fall below the competence floor and are excluded "
                   f"(sec:meth_gate). Every one of them is seed 2, and only at $k_v=12$; this "
                   f"must be stated, not omitted."),
              "",
              axis_table(rows, "k", lambda r: r["sigma"] == .5 and r["n"] == 4 and r["beta"] == 1.5,
                         "tab:axis_k", "Window size axis, at $K=4$, $\\sigma=0.5$, $\\beta=1.5$.", "$k_v$"),
              axis_table(rows, "n", lambda r: r["k"] == 12 and r["sigma"] == .5 and r["beta"] == 1.5,
                         "tab:axis_n", "Federation size axis, at $k_v=12$, $\\sigma=0.5$, $\\beta=1.5$.", "$K$"),
              axis_table(rows, "sigma", lambda r: r["k"] == 12 and r["n"] == 4 and r["beta"] == 1.5,
                         "tab:axis_sigma", "Contention axis, at $k_v=12$, $K=4$, $\\beta=1.5$.", "$\\sigma$"),
              axis_table(rows, "beta", lambda r: r["k"] == 12 and r["sigma"] == .5 and r["n"] == 4,
                         "tab:axis_beta", "Budget axis, at $k_v=12$, $K=4$, $\\sigma=0.5$.", "$\\beta$"),
              r"\section{Window Size} \label{sec:res_window}", "",
              note("A myopic rule is sufficient while the window is small and degrades as it "
                   "grows; the advantage changes sign between $k_v=8$ and $k_v=12$ on both "
                   "criteria independently.",
                   "Table tab:axis_k for the sweep; Table tab:checkpoint for the checkpoint "
                   "comparison; Figure crossover.pdf.",
                   "At $k_v=30$ two of three seeds favour the learned policy significantly and "
                   "one is indistinguishable. Report that, never the ratio of means."),
              "", checkpoint_table(),
              r"\section{Federation Size and Contention} \label{sec:res_scale}", "",
              note("The advantage reverses as agents are added. This is the boundary of the "
                   "contribution and the strongest argument for the future-work direction.",
                   "Tables tab:axis_n and tab:axis_sigma.",
                   "Two axes move together in the $K$ sweep and are separated only by the "
                   "$\\sigma$ column. Do not attribute the collapse to either alone."),
              "",
              r"\section{Where the Error Lands} \label{sec:res_reward}", "",
              note("The learned advantage is entirely on private-incident pairs, where it is "
                   "25x better than the myopic rule. Shared-shared pairs are solved by every "
                   "competent policy; only random errs there.",
                   "results/shd_by_class_200.json: 6 runs, 200 episodes, _best.pt. "
                   "learned 0.00002 / 0.00000, myopic 0.00051 / 0.00000, random 0.02302 / 0.00542.",
                   "DO NOT write the reward-alignment asymmetry. Ledger 1.3 claimed the learned "
                   "policy neglects unrewarded pairs; that is RETRACTED "
                   "(docs/FINDINGS_PAIR_CLASS_2026_09_02.md). Shared-shared error is 0.00000 "
                   "for both arms across 90,000 pair-observations."),
              "",
              r"\section{Transfer to a Realistic Evidence Regime} \label{sec:res_transfer}", "",
              note("RQ2, in three parts: the belief machinery carries to finite samples; the "
                   "policy does not, and the mechanism is measured; degrading the training "
                   "evidence recovers it.",
                   "Part 1 and part 2 are settled. Part 3 is TABLE AND FIGURE PENDING on the "
                   "answer-rate fleet (docs/FINDINGS_TRANSFER_2026_09_02.md).",
                   "All transfer measurements are at $k_v=8$; the headline cells are 20 and 30. "
                   "State the scale limit."),
              "",
              r"\section{The Price of This Formulation of Federation} \label{sec:res_federation}", "",
              note("Partitioning information, reward and optimisation costs little; a "
                   "communication-free positional convention is worse than not coordinating "
                   "at all; the learned policy beats both.",
                   "Table tab:federation; Figure federation.pdf; the $k_v=20$ SHD comparison.",
                   "Action rights stay partitioned in both arms, so this is not the cost of "
                   "decentralisation entire. Recovery rate saturates at $k_v=20$; the SHD "
                   "column is what separates the arms."),
              "", federation_table(),
              r"\section{Limits of Latent Attribution} \label{sec:res_attribution}", "",
              note("RQ4. Single-pair latents are attributed reliably; multi-pair latents at "
                   "exactly zero from two peers onward, unmoved by budget. The bound is "
                   "identifiability, not resources, and the missing experiment is a public good.",
                   "Tables tab:attribution and tab:coverage; Figure attribution_law.pdf.",
                   "Scoped section, not a chapter. The engine abstains rather than guessing -- "
                   "zero misattributions anywhere -- and that is what licenses the future-work "
                   "framing."),
              "", attribution_tables(),
              r"\section{Negative and Withdrawn Results} \label{sec:res_negative}", "",
              note("Claims made during the work and refuted by measurements queued to test them.",
                   "docs/RESULTS_LEDGER_2026_09_01.md section 6.",
                   "Half a page, table only. Evidence of how hard the results were pushed on."),
              ""]

    out = ROOT / "thesis/4 Results and Analysis.tex"
    out.write_text("\n".join(parts))
    print(f"wrote {out.name}: {len(parts)} blocks, {len(excl)} excluded runs "
          f"(seeds {sorted({r['seed'] for r in excl})})")


if __name__ == "__main__":
    main()
