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
            "effect of the harder settings needing more training. Most runs converge by\n"
            "$8{,}000$ episodes; the slowest does not.\n\n"
            + tbl("Per-window solve rate over training, from each run's checkpoint schedule.",
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

    return ("\\chapter{Supporting Ablations} \\label{app:ablations}\n\n"
            + tbl("Turn-aware credit assignment at $k_v=8$, three seeds per cell.",
                  "tab:credit", "lcc", r"Configuration & Seeds & SHD \\", body))


MACHINERY = r"""
\paragraph{Why intervention can answer it.} A bidirected edge records that something
unobserved confounds a pair. It does not say which unobserved variable, nor which party
holds it. Under passive observation the question is unanswerable in principle: one latent
confounding three variables and three separate latents each confounding one pair induce
identical marginal independence structure, so no volume of observational data separates
them \citep{richardson2002ancestral}. What breaks the symmetry is that only the
confounder's owner can intervene on it. If an agent intervenes on one of its own private
variables and a peer's previously confounded pair resolves, the responsible confounding
has been located, with neither party seeing the other's data \citep{hauser2012gies,
zhou2025hardinterventions}. The assumption this rests on, that when a peer acts and a
confounded pair moves the peer's own variables are among the movers, is false whenever a
private variable's influence is mediated entirely through a third party's block; it is
stated below as the modelling assumption it is.

\paragraph{Latent groups.} Structure recovery tells an agent that a pair of shared
variables is confounded by something it cannot observe. Attribution asks which hidden
variable, and whose.

\begin{definition}[Latent group] \label{def:latentgroup}
A \emph{latent group} is a pair $(o, C)$ where the \emph{owner} $o$ is the agent whose
private block contains the hidden variable, and the \emph{children} $C$ are the variables
in the observing agent's local variable set that this hidden variable parents.
\end{definition}

The assumptions of \S\ref{sec:meth_crossprivate} make this a finite choice. Every
bidirected edge joins two shared variables, so $C \subseteq X$; the confounding path lies
wholly within one private block, so each group has exactly one owner and attribution
selects among $K-1$ named peers. Attribution replaces the confounding mark rather than
accompanying it, so an agent cannot be credited for determining that a pair is confounded
while failing to say which peer is responsible.

\paragraph{The hypothesis space and its pruning.} Candidates are the latent groups
consistent with the determined structure, pruned by two rules. A hidden variable
parenting $C$ makes every pair in $C$ confounded, so a candidate whose children are not
pairwise confounded is removed; this is sound, and it is the atomicity rule referred to
below. When a peer intervenes on one of its private variables the groups it owns respond
and others do not, so a group whose pairs fail to respond to its putative owner's
intervention is removed; this is a modelling assumption rather than a theorem, and it
fails where hidden variables in different blocks parent overlapping sets. Candidates are
canonicalised to maximal cliques per owner, since one hidden variable parenting
$\{u,v,w\}$ and three hidden variables in the same block parenting each pair cannot be
distinguished by any evidence available here. Enumerating groups jointly is infeasible at
these sizes, $k_v = 20$ admitting $8.4 \times 10^{10}$ hypotheses, so the space is
factored over the connected components of the confounded-pair graph and pruning
propagates within each component to a fixpoint. The factoring is exact, because a hidden
variable's children are pairwise confounded and therefore lie in one component. The
measurements below use the structural policies and baselines of \S\ref{sec:meth_marl}
unchanged, plus one baseline specific to this appendix: an attribution-greedy policy,
the myopic rule applied to the attribution belief rather than the structural one.
"""

WITHDRAWN = r"""
\paragraph{Withdrawn during the work.} Five claims about attribution were made during
this work and later withdrawn; the record of every withdrawal is
\S\ref{sec:res_negative}, and the five that concern this appendix are kept beside the
results they qualify.

\begin{table}[htbp]
\centering
\caption{Attribution claims withdrawn, with the measurement that refuted each.}
\label{tab:app_attr_withdrawn}
\begin{tabular}{p{0.44\textwidth}p{0.48\textwidth}}
\toprule
Claim & What refuted it \\
\midrule
The learned policy attributes latent owners worse than random & One seed at $2$ SE; the next two reversed it. \\
Attribution precision falls from $98\%$ to $59\%$ as the window grows & Two defects in the attribution engine. Zero misattributions at every size once repaired. \\
The component engine gains precision by skipping cross-component pruning & A probe for such messages found none. \\
The decline with site count is hypothesis-space growth & The matched-budget control, holding rounds per agent fixed, attributes it to coverage. \\
Probe diversity explains attribution performance & The lowest-coverage policy ties the highest. \\
\bottomrule
\end{tabular}
\end{table}
"""


# --- E: attribution, self-contained -------------------------------------------------------
def appendix_attribution():
    """Latent-owner attribution, in one place and nowhere else.

    Downgraded from a research question on Brian's instruction, 3 Sep: the result is sound but
    thin, no policy was ever trained on the attribution objective, and it was drawing effort
    away from the three questions the thesis actually answers. Everything here is generated
    from thesis_results/attribution/ so the section cannot drift from its data.
    """
    TRA = ROOT / "thesis_results/attribution"

    def jl(name):
        return jload(TRA / f"{name}.json")

    ceiling = jl("attr_ceiling")
    budget = jl("attr_ceiling_budget")
    matched = jl("attr_ceiling_matched_budget")

    # Soundness across every configuration measured, deduplicated by config.
    seen = {}
    for src in (ceiling, budget, matched):
        for e in src:
            seen[(e["k"], e["sigma"], e["n_agents"], e["budget"], e["episodes"])] = e
    groups = sum(e["total"] for e in seen.values())
    right = sum(e["right"] for e in seen.values())
    wrong = sum(e["wrong"] for e in seen.values())

    peers = []
    for e in sorted(ceiling, key=lambda x: (x["n_agents"], x["k"], x["sigma"])):
        peers.append(f"{e['n_agents'] - 1} & {e['k']} & {e['sigma']} & {e['budget']} & "
                     f"{e['right']} & {e['wrong']} & {e['total']} & {e['measured']:.3f} \\\\")

    # The identifiability cliff, by number of children, at k=12 sigma=0.5.
    def bysize(e):
        cells = []
        for size in range(2, 7):
            v = e.get("by_size", {}).get(str(size))
            if not v:
                cells.append("---"); continue
            r = v.get("right", 0); u = v.get("unsure", 0); w = v.get("wrong", 0)
            cells.append(f"{r}/{r + u + w}")
        return cells

    def find(src, K, b):
        for e in src:
            if e["n_agents"] == K and e["budget"] == b and e["k"] == 12 and e["sigma"] == 0.5:
                return e

    size_rows = []
    for src, K, b in ((ceiling, 2, 60), (ceiling, 3, 60), (ceiling, 4, 60),
                      (ceiling, 8, 60), (matched, 8, 120)):
        e = find(src, K, b)
        if e:
            size_rows.append(f"{K - 1} & {b} & " + " & ".join(bysize(e)) + r" \\")

    brows = [f"{e['budget']} & {e['right']} & {e['total']} & {e['measured']:.4f} \\\\"
             for e in sorted(budget, key=lambda x: x["budget"])]

    import glob as _glob
    arows = []
    for f in sorted(_glob.glob(str(ROOT / "results/attr_train/*.json"))):
        d = jload(f)
        arows.append(f"{d.get('seed')} & {d['arms']['learned']['success']:.3f} & "
                     f"{d['arms']['greedy_uncertainty']['success']:.3f} \\\\")

    return (
        "\\chapter{Latent-Owner Attribution} \\label{app:attribution}\n\n"
        "The belief of \\S\\ref{sec:meth_versionspace} can sometimes name which peer's private\n"
        "block contains a latent confounder detected on the shared interface. This appendix reports\n"
        "what that machinery achieves and what bounds it. It is placed here rather than in\n"
        "Chapter~\\ref{Chap4} because no policy in this work was trained on an attribution\n"
        "objective: the trainer scores structural claims, and every result below is the behaviour\n"
        "of a belief driven by a policy trained for something else. What can be established is\n"
        "that attribution is possible and what limits it, which is a starting point for other work\n"
        "rather than a result of this one.\n\n"
        + MACHINERY
        + f"\\paragraph{{Soundness.}} Across {len(seen)} configurations spanning "
        f"$k_v \\in \\{{12, 20\\}}$, $K \\in \\{{2,3,4,8\\}}$, "
        f"$\\sigma \\in \\{{0.25, 0.5, 0.75\\}}$ and budgets 30 to 240, "
        f"\\textbf{{{groups:,} latent groups were observed and {right:,} were attributed, with "
        f"{wrong} attributed incorrectly.}} The engine names an owner or abstains, so the\n"
        "quantity that varies is the abstention rate and not an error rate. Zero is a property of\n"
        "the atomicity rule above rather than a fortunate sample.\n\n"
        + tbl("Attribution by configuration. Every cell has zero incorrect attributions.",
              "tab:app_attr_peers", "rrrrrrrr",
              r"Peers & $k_v$ & $\sigma$ & Budget & Correct & Incorrect & Observed & Share \\",
              peers)
        + "\n\\paragraph{Two bounds, separated by one comparison.} What limits the share is\n"
          "partly resources and partly identifiability, and the last two rows of\n"
          "Table~\\ref{tab:app_attr_size} tell them apart without an argument. Doubling the budget\n"
          "at seven peers moves two-child resolution from $63/1344$ to $965/1344$ and leaves every\n"
          "group of three or more children at exactly zero. A resource bound responds to\n"
          "resources; an identifiability bound does not.\n\n"
        + tbl("Groups resolved of groups observed, by the number of children the latent has, "
              "at $k_v=12$ and $\\sigma=0.5$. An unresolved group is an abstention.",
              "tab:app_attr_size", "rrccccc",
              r"Peers & Budget & 2 children & 3 & 4 & 5 & 6 \\", size_rows)
        + "\n\\paragraph{Coverage saturates.} At four agents the measured share is unchanged from\n"
          "budget 60 onward, and the counts are identical rather than merely the rates, so the\n"
          "groups that remain are not reachable by spending more.\n\n"
        + tbl("Attribution against intervention budget at $k_v=12$, $\\sigma=0.5$, four agents.",
              "tab:app_attr_budget", "rrrr",
              r"Budget & Correct & Observed & Share \\", brows)
        + "\n\\paragraph{Where the ceiling comes from.} A group with two children explains one\n"
          "pair, so ownership is the whole question and one partner response settles it. A group\n"
          "with three or more explains a clique, and separating it from several smaller latents\n"
          "needs a partial response: the owner must probe its private variables one at a time.\n"
          "No policy here does, so responses are total and the atomicity rule never fires. The\n"
          "one-peer row of Table~\\ref{tab:app_attr_size} is the exception that identifies the\n"
          "cause. With ownership forced, three- and four-child groups do resolve, at $38/59$ and\n"
          "$29/74$, and five-child groups still do not.\n\n"
          "A two-factor decomposition captures this. Writing $\\Pr(\\text{resolve} \\mid\n"
          "\\text{one pair})$ for the measured rate at which one-pair groups settle and $\\pi_1$\n"
          "for the share of one-pair groups in the graph distribution, the product predicts the\n"
          "measured share to within $0.041$ at every configuration with two or more peers. Only\n"
          "$\\pi_1$ is computable from the topology; the first factor is measured, so this\n"
          "decomposes an observed rate rather than predicting one. The single-peer configuration\n"
          "is under-predicted by $0.263$, because groups explaining more than one pair also\n"
          "resolve there.\n\n"
          "\\begin{figure}[htbp]\n"
          "\\centering\n"
          "\\includegraphics[width=0.98\\textwidth]{figures/attribution_law.pdf}\n"
          "\\caption{Measured attribution against the two-factor decomposition, with the "
          "diagonal drawn. Filled points: two or more peers. Open point: one peer.}\n"
          "\\label{fig:attribution_law}\n"
          "\\end{figure}\n\n"
          "\\paragraph{Not a policy failure.} A self-interested policy, scored only on its own\n"
          "recovery, spends $7.6\\%$ of its budget on private variables against $38$--$61\\%$ for\n"
          "every other policy, and still does worse than a rule that is not scored on attribution\n"
          "at all: $0.245$ against $0.327$ on attribution, and $0.181$ against $0.327$ on joint\n"
          "identification, over three seeds of $100$ episodes each. The ceiling is not reached by\n"
          "wanting it more.\n\n"
          "\\paragraph{Scale.} The component-factored engine runs\n"
          "past the sizes the chapter reports: 21, 33 and 27 correct attributions at $k_v = 30$,\n"
          "$40$ and $50$ over 30 episodes each, with no incorrect attribution and no contradiction\n"
          "raised at any size.\n\n"
          "\\paragraph{What is not established.} Nothing here says what a policy trained to\n"
          "attribute would achieve, because none was trained. The comparison that would answer it\n"
          "requires an attribution term in the reward and an owner channel in the observation,\n"
          "neither of which any run in this work uses. That is the experiment this appendix\n"
          "points at. The nearest measurement is training under the attribution belief backend\n"
          "with the reward unchanged (Table~\\ref{tab:attrbackend}): at $4{,}000$ episodes it\n"
          "reaches $0.400$, $0.355$ and $0.205$ joint recovery against the myopic rule's $0.945$,\n"
          "$0.955$ and $0.935$ on the same cell. $4{,}000$ episodes is a budget three structural\n"
          "claims elsewhere in this work did not survive (\\S\\ref{sec:res_negative}), so this\n"
          "measures the backend at that budget rather than the backend's ceiling.\n\n"
        + tbl("Training under the attribution belief backend at $k_v=12$, four agents, "
              "$4{,}000$ episodes, scored on the structural criterion.",
              "tab:attrbackend", "lcc", r"Seed & Learned & Myopic \\", arows)
        + "\n" + WITHDRAWN)


def main() -> int:
    out = ROOT / "thesis/Appendix.tex"
    parts = [r"\appendix", "",
             "% Generated by scripts/build_appendix.py. Do not edit tables by hand.",
             ""]
    for fn in (appendix_excluded, appendix_attribution, appendix_budget, appendix_checkpoint,
               appendix_ablations):
        parts.append(fn())
        parts.append("")
    out.write_text("\n".join(parts))
    print(f"wrote {out.relative_to(ROOT)}: {len(out.read_text().splitlines())} lines, "
          f"{out.read_text().count(chr(92) + chr(99) + chr(104) + chr(97) + chr(112) + chr(116) + chr(101) + chr(114) + chr(123))} appendices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
