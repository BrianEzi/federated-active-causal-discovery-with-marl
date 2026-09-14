# Overnight sync with the other agent

Fetches every 10 minutes. Code changes (ma/, cb/, scripts/) are NOT auto-merged while
matched training runs are queued -- taking them mid-experiment would silently void the
matched-pair design. Anything flagged HELD below needs a human to merge deliberately.

## 02:41 — 5 new commit(s) from them
  - 5b3e473 results: the dial survives a fair greedy; the transition fix does not rescue eight agents
  - 384b82b Merge branch 'explore/constraint-based' of https://github.com/BrianEzi/federated-active-causal-discovery-with-marl into explore/constraint-based
  - cfbc0f7 feat: turn-aware credit, and a re-scorer that rebuilds the env from a run's own config
  - 6d2d54d Merge branch 'explore/constraint-based' of https://github.com/BrianEzi/federated-active-causal-discovery-with-marl into explore/constraint-based
  - 9a4e7f2 results: the full 14-arm re-score at the graded greedy bar

**HELD — touches experiment code, not merged automatically:**
  - ma/policy.py
  - scripts/ma_train.py
  - scripts/rescore_from_config.py
  - scripts/run_dial_fairbar.sh

  Merge by hand once the queued runs are done: `git pull --rebase origin explore/constraint-based`

