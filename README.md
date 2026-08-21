# Federated Active Causal Discovery with Multi-Agent RL

MSc thesis. Several agents each see part of a causal system, choose interventions to run, and
must recover the structure — **without a central server and without sharing private data.**
The difficulty the setting creates: a variable one agent cannot see can confound two it can,
so an agent must sometimes spend a move that helps only its partner.

## Start here

| you want | read |
|---|---|
| what is true right now | **`docs/STATE_OF_TRUTH.md`** — established, retracted, open |
| how scoring works | `docs/SCORING.md` |
| what every term means | `docs/GLOSSARY.md` |
| the current protocol | `docs/TURN_BUDGET_SPEC.md` |
| references | `docs/BIBLIOGRAPHY.md` |

**Do not cite a number straight out of `docs/logs/`.** Those are chronological and contain
claims withdrawn later. Check `STATE_OF_TRUTH.md` first.

## Layout

    sa/         single agent: graphs, BGe scoring, exact posterior (subset DP), oracle, PPO
    ma/         two agents: windows, regime scoring, confounding, environment, policy
    scripts/    only what reproduces a number we cite
    tests/      570 tests
    docs/       authoritative set; logs/ and archive/ behind warnings
    legacy/     superseded code, kept not deleted -- see legacy/README.md

`legacy/ma_v1/env.py` looks like dead code and **must not be deleted**: it generated the
fixture that validates the subset DP to 1e-10, and regenerating it from current code would
make the check circular.

## Reproduce the headline result

    pip install -r requirements.txt
    PYTHONPATH=. python -m scripts.ma_seed_batch --arm tb_clamp --seeds 0-9 --jobs 5 \
        --disclose_regime --turn_order round_robin --budget 10 --clamp_only \
        --train_episodes 2000 --eval_episodes 150

About 40 minutes on 5 cores. Expect the learned policy near **0.55** against a random floor
near **0.38**, and — the actual finding — roughly **82% of its clamps aimed at its own private
node**, against a chance rate of 25% and a myopic-greedy rate of 19%.

That gap is the result: the agents learn to spend moves on the intervention that helps their
partner rather than themselves, from a shared reward alone, with no communication beyond one
bit per round.

## Tests

    python -m pytest tests/ -q
