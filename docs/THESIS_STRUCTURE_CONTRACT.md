# Thesis structure contract

Author: Brian. Recorded 2 Sep 2026. **This governs every chapter and overrides any
earlier structural note in `docs/`.** Agent C writes to it; agents A and B generate
data and tables to fit it.

---

## 1. The governing rule

> **One research question drives one section of Results. Each core experiment answers
> one research question.**

Nothing else determines the shape of Chapter 4. Sections are not named after axes,
after experiments, or after findings. They are named after the question they settle.
An experiment that answers no research question is appendix material or is cut.

Four research questions, therefore four Results sections, plus one for what did not
survive:

| § | Research question | Core experiment |
|---|---|---|
| 4.1 | **RQ1** Federated active recovery, and how the advantage varies with window size, federation size, contention, budget | The four-axis sweep (20 cells × 3 seeds, 12,000 episodes) |
| 4.2 | **RQ2** Transfer to a realistic evidence regime | Oracle → sampled evidence; partial oracle at answer rate ρ |
| 4.3 | **RQ3** The price of this formulation of federation | The federation ladder: centralised → federated, matched cells |
| 4.4 | **RQ4** Limits of latent attribution | Attribution audit over the sweep's episodes |
| 4.5 | Negative and withdrawn results | — |

§4.5 is not a research question. It exists because the work retracted claims it had
already drafted, and reporting that is a mark of the thesis's honesty rather than a
concession. It stays last.

---

## 2. Chapter charters

Each chapter has exactly one job. A sentence that does two chapters' jobs at once is
in the wrong chapter.

### Chapter 2 — Background and Related Work
**Job: make the thesis self-contained, then place this work precisely between the
literatures it sits between, and say why the gap is interesting.**

A walkthrough of everything the reader needs and nothing they do not. It ends by
locating the contribution: not "little work exists on X", but *here is the active
causal discovery literature, here is the federated/vertically-partitioned literature,
here is what neither of them can currently answer, and here is why anyone should
care*. The gap is argued, not asserted.

### Chapter 3 — Methodology
**Job: explain plainly how things were implemented, and what the experimental setup is.**

This chapter owns **all** setup exposition, without exception:

- the environment, belief tracker, policy, federated optimiser;
- the metrics and what they normalise by;
- the evaluation protocol — paired episodes, seeds, error bars;
- the baselines and the competence gate;
- the two conventions every number depends on: **checkpoint selection** and
  **action selection at evaluation** (sampling at temperature 1);
- `\S`**Experiments**: the axes, the cells, the ladder, the budgets — the full
  description of every experiment run, stated once.

If a reader needs to know it *before* they can read a number, it belongs here.

### Chapter 4 — Results and Analysis
**Job: report what the experiments show, and analyse it.**

No experimental setup. None. Not a recap "for convenience", not a reminder of how many
seeds, not a restatement of the protocol. Each section cites Chapter 3 (`\S\ref{...}`)
and proceeds directly to the result. If a section cannot be understood without a
paragraph of setup, the fix is a clearer Chapter 3, not a paragraph in Chapter 4.

Analysis lives here, and it is extensive — this is where the numbers are interrogated,
where a confound is chased down, where a boundary is drawn around a claim. By the end
of Chapter 4, the analysis is *finished*. Chapter 5 has no analytical work left to do.

### Chapter 5 — Discussion
**Job: tie it together and say what it means.**

Not more analysis. Synthesis: what the four answers amount to taken together, what
they imply for federated causal discovery as a proposition, where the argument is
load-bearing and where it is thin, how it sits against the related work of Chapter 2.
The insight chapter.

> **Ownership: this chapter is Brian's exclusively.** So is much of the analysis in
> Chapter 4. Agents do not draft Chapter 5. Agents do not draft interpretive prose in
> Chapter 4 beyond stating what the data shows and where it stops. Where an agent
> believes an interpretation is warranted, it goes in a `docs/FINDINGS_*.md` note for
> Brian to use or discard — never straight into the chapter.

### Chapter 6 — Conclusion
**Job: close the loop, briefly.**

Concise. Links back to the Introduction — the questions asked there, answered here.
A short statement of what succeeded, a short statement of limitations, a short
statement of future work. No new material, no new numbers, no re-argument.

---

## 3. What this means for the current draft

Chapter 4 currently has eight sections organised by experiment. They collapse as
follows.

| Current section | Destination |
|---|---|
| §4.1 The Sweep | §4.1 RQ1 — becomes the section |
| §4.2 Window Size | fold into §4.1 (an axis, not a question) |
| §4.3 Federation Size and Contention | fold into §4.1 (two axes, not a question) |
| §4.4 Where the Error Lands | fold into §4.1 as the mechanism behind the contention gradient |
| §4.5 Transfer to a Realistic Evidence Regime | §4.2 RQ2 |
| §4.6 The Price of This Formulation of Federation | §4.3 RQ3 |
| §4.7 Limits of Latent Attribution | §4.4 RQ4, narrowed |
| §4.8 Negative and Withdrawn Results | §4.5, stays last |

Moving out of Chapter 4 entirely:

- the Measurement Protocol section — **cut**; Chapter 3 `sec:meth_eval` and
  `sec:meth_paired` own it;
- every "60 runs, 20 cells, three seeds" style preamble — into `sec:meth_ladder`;
- the checkpoint and sampling conventions — into Chapter 3, stated once, cited after.

Two other structural repairs, both already identified:

1. **RQ1 and RQ2 overlap in the Introduction.** RQ1's clause about the evidence
   spectrum belongs wholly to RQ2. One question, one section, one boundary.
2. **The Contributions list in `1 Introduction.tex` is stale.** It still claims BGe
   scoring, a two-agent partition, a 3.5× factor and an 82–91% clamp allocation —
   none of which the current work reports. It must be rewritten against `CLAIMS.md`
   before submission.

---

## 4. RQ4 is deliberately narrow

Attribution is a research question, but a small one. The claim it may make is that
attribution **is possible**, and under precisely which conditions it is achievable —
no more. No agent was ever trained on the attribution objective, so nothing may be
said about what a trained attributor would achieve. The section leads into future
work; it does not carry the thesis. Thin results are not to be over-explained into
looking thick.

Any attribution claim beyond "possible, under these conditions" needs the same weight
of evidence as every other claim in the thesis, and does not currently have it.

---

## 5. The placement test

Before writing a sentence, ask which of these it is:

- **"Here is how it works / how it was set up"** → Chapter 3.
- **"Here is what happened, and here is what that number does and does not support"**
  → Chapter 4.
- **"Here is what it all means"** → Chapter 5, Brian's.
- **"Here is what we asked and what we found, in three lines"** → Chapter 6.
- **"Here is what the reader must already know to follow any of it"** → Chapter 2.

A sentence that fits two of these is split. A sentence that fits none is cut.

---

## 6. Numbers

`thesis_results/CLAIMS.md` is the single source for every number that enters the
thesis, and it carries the `MUST NOT` boundaries for each claim. It is generated from
data by `scripts/build_claims.py`; nothing in it is edited by hand except those
boundary lines. A number that is not in `CLAIMS.md` does not go in the thesis.

Note in particular: **a result file's own `global_hard_shd` field is not what the
thesis reports** and differs from it by up to 300× on the same seed. Every structural
number comes from `scripts/global_shd_paired.py`.
