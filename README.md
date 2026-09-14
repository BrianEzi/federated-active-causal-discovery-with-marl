# Federated Active Causal Discovery via Multi-Agent Reinforcement Learning

Code, data and analysis for the MSc dissertation of the same name.

Several parties each hold measurements over an overlapping set of variables. The causal
structure they care about crosses their boundaries, no party observes all of it, and none
may share raw data. Structure that observational data cannot settle has to be settled by
intervening, and interventions are budgeted. The question is whether such parties can learn,
independently and without a coordinator, which experiments to run.

The system poses this as a decentralised partially observable Markov decision process.
Each agent holds a window of variables, maintains a version space over the ancestral marks
between pairs in that window, and chooses interventions under a policy trained by
independent PPO. Beliefs held at different sites compose by intersection, so a global
structure is recovered without data, parameters or gradients crossing a boundary.

## What the dissertation reports

Three questions, with the evidence in `thesis_results/CLAIMS.md`:

- **Does learned experiment selection beat myopic uncertainty targeting?** Yes, in sixteen
  of eighteen configurations once trained to convergence. At a third of that training
  budget it wins two. The training budget, not any swept structural parameter, decides.
- **Does it survive evidence estimated from finite samples?** Only if training itself
  withholds answers. Policies trained under a partial oracle transfer; policies trained on
  free answers do not.
- **What does federating cost?** Nothing detectable at the cells measured, reported as an
  equivalence bound rather than as a failure to find a difference.

The advantage is a scarcity effect. It is largest where intervention rounds are few, and
under scarcity what the policy has learned is to allocate a shared budget across windows,
which is something no per-window rule attempts.

## Layout

    ma/            environment, topology, SCM, policies, baselines, evaluation
    cb/            the constraint-based belief: version space, CI tests, orientation,
                   latent-owner attribution
    crosscheck/    an independent dynamic-programming belief, used to validate cb/
    legacy/        the retired first-generation implementation, kept because parts of the
                   test suite check the current belief against it, so a shared bug cannot
                   hide in both
    scripts/       training, evaluation, analysis and the generators that build every
                   reported table and figure
    tests/         489 tests; `pytest -m "not slow and not perf"` for the fast subset
    results/       measurement output as JSON, one file per evaluated cell
    submission/    the curated subset the dissertation cites, with checkpoints and a
                   manifest that can be re-hashed
    thesis_results/  CLAIMS.md, the number source; RETRACTIONS.md, the correction record

## Documentation

    docs/ARCHITECTURE.md   how an episode runs and where each component sits
    docs/BELIEF.md         the version space, why it composes, and what it cannot do
    docs/EVALUATION.md     the paired protocol, checkpoint conventions, and the metrics
    docs/REPRODUCING.md    how to rebuild any reported number from this repository
    docs/DATA.md           what is in results/ and submission/, and how provenance is kept

## Getting started

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest -m "not slow and not perf"

`docs/REPRODUCING.md` walks one reported number from raw runs to the figure it appears in.

## On reading the numbers

`thesis_results/CLAIMS.md` is the only source for a reported value. Every claim there
carries its boundary, and many carry explicit MUST NOT lines recording a reading the data
does not support — usually because an earlier version of the analysis made exactly that
mistake. `thesis_results/RETRACTIONS.md` lists nineteen claims that were withdrawn or
corrected during the work.

Numbers recorded by a training run in its own result file are not what the dissertation
reports. Policies are re-scored after the fact by `scripts/global_shd_paired.py` under a
seeded, paired protocol, and the two can differ substantially. `docs/EVALUATION.md`
explains why.

## Provenance

This branch is the reviewer-facing repository. The full working history is preserved on the
`pre-squash-submission` tag: 755 commits that are not on this branch, the complete results
tree including every training checkpoint, and the running experiment log.

The tag carries about 2.6 GB of checkpoints, so a default clone fetches all of it. To take
only the reviewer-facing branch:

    git clone --single-branch --branch main <url>

The dissertation text is versioned separately and is not in this repository.
