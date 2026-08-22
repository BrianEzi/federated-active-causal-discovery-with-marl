# HANDOVER: convert `ma/env.py` to n agents

**Read this whole file before touching any code. Follow it in order. Do not skip the
"stop and check" steps.**

## 0. What you are doing, in one sentence

`ma/topology.py` already supports n agents (done, tested, on this branch). `ma/env.py`
still hardcodes exactly 2 agents named `"A"` and `"B"`. Your job: make `ma/env.py` (and
everything that calls it) work for 3 and 5 agents too, WITHOUT breaking the existing 2-agent
behaviour or any existing test.

## 1. Ground rules (read these twice)

- **Branch**: `feat/n-agent-topology`. Already exists, already pushed. Work on it. Do NOT
  touch `main`. Do NOT merge anything. Do NOT touch `cluster/` or ssh to `myriad` — a
  training job may be running there, leave it alone.
- **No compatibility shims.** If you find yourself writing "if old style then X else Y",
  stop — that is a shim. This project has been bitten twice by shims that quietly kept old
  behaviour "working" while silently meaning the wrong thing (a budget that changed meaning
  without anyone noticing; a rule that could structurally never fire). Change the call site
  instead. Every place that hardcodes `"A"`/`"B"` gets fixed, not wrapped.
- **Comments**: only explain WHY, never WHAT. Do not write "# loop over agents" above a for
  loop. Do write "# n-1 partner signals, not the module constant, because n > 2 here" when
  that is the actual reason.
- **Do not add scope.** You are NOT building per-block confounding subsets (the "S_r"
  thing). That is a separate, harder, already-tracked task for topologies where ONE AGENT
  has more than one private node. `ma/env.py` already REFUSES those topologies with a raised
  error in `__init__` — leave that guard alone, do not remove it, do not work around it.
  Your job only needs to work for topologies where every agent has exactly ONE private node
  (this covers the 3-agent and 5-agent test shapes below). If a test needs 2 private nodes
  per agent, that test is out of scope — skip it, note it, move on.
- **After every file you finish, run the full test command in section 6.** If it goes red,
  you broke something — fix it before moving to the next file. Do not batch up five files
  and then discover 40 failures.
- **Commit in small pieces**, one logical change each, with a message that says WHY (see
  section 8 for the format this project uses). Push after each green commit.

## 2. The size of the problem (already measured, do not re-derive this)

Two kinds of hardcoding exist, and they are different sizes:

**A. Inside `ma/env.py` itself** — the `AGENTS = ("A", "B")` module constant, used ~30
times, plus TWO genuinely hardcoded 2-agent spots that are not simple find-replace:

```python
def step(self, action_a: int, action_b: int) -> StepResult:      # exactly 2 positional args
    actions = {"A": int(action_a), "B": int(action_b)}
    ...
other = "B" if name == "A" else "A"                               # "the other agent" — only
                                                                    # means something at n=2
```

This second pattern (`other = "B" if name == "A" else "A"`) appears TWICE in `ma/env.py`:
once around line 389 (disclosing the partner's shared-node target) and once in
`_signal_onehot` around line 472 (the partner's broadcast signal). At n agents there is no
single "other" — there are n-1 others. Both spots need to become a loop or a per-partner
structure, not a single value.

**B. Outside `ma/env.py`** — these files import `AGENTS` from `ma.env` and loop over it
(fixing the constant fixes them for free), OR hardcode `"A"`/`"B"` literally (these need
individual fixes):

```
grep -rln "from ma.env import.*AGENTS" --include=*.py .
  ma/evaluate.py
  ma/policy.py
  scripts/ma_gate2_collision.py
  scripts/ma_gates2.py
  scripts/ma_graph_examples.py
  scripts/ma_train.py
  tests/ma/test_env.py
  tests/ma/test_greedy_tiebreak.py
  tests/ma/test_metric_reachability.py
  tests/test_env_turns.py
  tests/test_env_turn_budget.py

grep -rln '"A"\]\|\["A"\|"A": \|== "A"\|== "B"' --include=*.py ma/ sa/ scripts/
  (excluding ma/env.py, ma/topology.py, which you are already fixing)
  ma/baselines.py
  ma/coordination.py
  ma/evaluate.py
  ma/policy.py
  scripts/ma_gate2_collision.py
  scripts/ma_gates2.py
  scripts/ma_graph_examples.py
  scripts/ma_graph_render.py
```

**This means the task is NOT "edit one file". It is env.py plus at minimum
`ma/evaluate.py`, `ma/policy.py`, `ma/baselines.py`, `ma/coordination.py`.** Budget your
time accordingly. If you only have time for env.py itself, STOP after section 4 and write
up what's left in section 9's format rather than half-editing a downstream file.

Also, `env.step(...)` is called positionally (2 args) from these files — every one of them
needs updating when you change the signature:

```
ma/evaluate.py
ma/policy.py
scripts/ma_gate2_collision.py
scripts/ma_gates2.py
scripts/ma_graph_examples.py
tests/ma/test_belief_crosscheck.py
tests/ma/test_env.py
tests/ma/test_evaluate.py
tests/ma/test_metric_reachability.py
tests/test_env_turns.py
tests/test_env_turn_budget.py
```

(Do NOT touch `tests/sa/test_env.py`, `tests/sa/test_policy.py`, `tests/test_env_dp.py`,
`tests/test_sampling_oracle.py`, `tests/test_uncertainty.py` — those call the UNRELATED
single-agent `sa/env.py`, different module, different signature, not your problem.)

## 3. The decision that is ALREADY MADE — do not re-litigate it

`docs/N_AGENT_REFACTOR_SPEC.md` section 4 was written and APPROVED by the student on
2026-08-22. Read it before starting. The relevant parts, so you don't have to hunt:

- **Agent identity**: agents become integers, not letters. `ma/topology.py` already made
  this switch (`topology.agents` is `(0, 1)`, `(0, 1, 2)`, etc). `ma/env.py` should follow
  suit: **use integers `0..n-1` as the keys for every dict this class owns**
  (`self.windows`, `self.forfeits`, `self.clamps_private`, `self.signals`, etc), not the
  strings `"A"`/`"B"`. This is a bigger change than keeping strings, but it is the change
  the spec already commits to, and generating fresh letters (`"A", "B", "C", ...`) would
  just be a second, incompatible naming scheme sitting next to `ma/topology.py`'s integers
  for no reason.
- **`step()` signature**: becomes `step(self, actions: Dict[int, int]) -> StepResult`,
  mapping agent index to action index. Every call site passes a dict now, e.g.
  `env.step({0: idx_a, 1: idx_b})` instead of `env.step(idx_a, idx_b)`.
- **Turn order**: `active_agent()`'s round-robin already generalises
  (`self.topology.agents[self.round % self.topology.n_agents]`); random already works with
  any-length tuple. Small change, not a redesign.
- **The "other agent" spots**: become a loop over `self.topology.agents` excluding `name`,
  not a single value. E.g. for the shared-target disclosure, an agent needs to know about
  EVERY OTHER agent's shared move this round, not just one partner's — so `self.disclosed`
  needs to become per-partner (a dict of dicts, or an array indexed by other-agent), and
  `_signal_onehot` needs one one-hot block PER OTHER AGENT, concatenated, so the
  observation width grows as `O(n)`. This changes the observation SHAPE, which means any
  saved policy checkpoint trained at 2 agents will not load into a 3-agent env — that is
  expected and fine, do not try to prevent it.
- **The clean rule**: already `any(hidden clamped)`, already correct for the n-agent,
  one-private-node-each case you are building for. Do not touch its logic, it works.

If you hit a design question NOT answered by the paragraphs above or by
`docs/N_AGENT_REFACTOR_SPEC.md`, STOP. Do not guess. Write the question down in
`docs/logs/MA_BUILD_LOG.md` under a `**[BLOCKED]**` heading with the exact question, commit
that, and stop working on that part. Move to a different part if there is one, or stop
entirely and report back.

## 4. Order of work

1. **`ma/env.py` internals**: switch `AGENTS` usage to `self.topology.agents` (integers),
   fix `step()`'s signature, fix both "other agent" spots to loop over all partners, fix
   `active_agent()`. Run `PYTHONPATH=. python -m pytest tests/ -q` — expect many failures,
   that's fine, this is the biggest single edit. Do not chase every failure yet; get the
   file itself internally consistent first (it should at least IMPORT and construct an env
   without crashing).
2. **`ma/evaluate.py`**: fix its `AGENTS`/`"A"`/`"B"` usage and its `env.step(...)` calls.
   Run the test command again.
3. **`ma/policy.py`**: same.
4. **`ma/baselines.py`**: same.
5. **`ma/coordination.py`**: same. (Check first whether `coordination.py` is even on the
   import path for anything you're changing — it may be dead weight from before the
   turn-budget rewrite. `grep -rln "from ma.coordination"` and look.)
6. **Every file in the two grep lists in section 2** that you haven't already hit.
7. **A smoke test you write yourself**: construct a 3-agent, 1-private-node-each topology
   (`Topology(private=((0,),(1,),(2,)), exposed=(3,4,5))`), build a `TwoAgentEnv` (rename
   nothing yet — that class is still called `TwoAgentEnv`, renaming it is out of scope,
   don't do it) with it, call `.reset()` then `.step({0: 0, 1: 0, 2: 0})` a couple of times,
   assert it doesn't crash and `StepResult` has 3 entries in its per-agent dicts. Put this
   test in `tests/ma/test_env.py` near the other tests in that file, following their style.
8. **Full suite green.** Section 6.

## 5. Things that will bite you specifically

- `_result()` and anywhere `beliefs={n: ... for n in AGENTS}` appears — these build the
  `StepResult` returned to the caller, and every one of them needs to iterate the new agent
  set, not the module constant.
- `self.disclosed` is currently a flat array per agent (one partner's shared targets). At
  n>2 it needs a shape per OTHER agent. Decide the shape once, document it in a one-line
  comment at the field definition, and use it consistently — do not let two methods
  disagree about the shape of the same field, that produces a silent wrong-answer bug that
  passes green tests (this exact class of bug has happened twice already in this project).
- `_signal_onehot`'s docstring says "the observation width does not change with the flag,
  so a checkpoint stays loadable" — that guarantee is about the `disclose_signals` FLAG, not
  about agent COUNT. It is fine and expected that a 2-agent checkpoint won't load into a
  3-agent env. Do not try to preserve cross-n loadability, nobody asked for that and it
  would be over-engineering.
- The `done_bit` and regime-bit logic loops over `AGENTS` too — same fix, same care.

## 6. The test gate — run this after every file, non-negotiable

```
cd "C:/Workspace/MSc Project/.claude/worktrees/single-agent-clean"
PYTHONPATH=. python -m pytest tests/ -q
```

**Before you start, this must say `470 passed`.** If it doesn't, STOP — something is
already broken and it is not your job to fix it silently; report it first.

**When you are done, it must again say all tests passed**, plus whatever new tests you
added in step 7 of section 4. If a pre-existing test now fails, that test encoded the OLD
2-agent-only behaviour and needs updating to the new API (e.g. `env.step(a, b)` becoming
`env.step({0: a, 1: b})`) — update the test, do not delete it, do not weaken its assertion
to make it pass.

Use `-m "not slow"` for a faster loop while iterating (~2 min instead of ~4), but the FINAL
check before any commit must be the full command above with nothing deselected.

## 7. What "done" looks like

- `470+` tests pass (470 plus whatever you added).
- A 3-agent, 1-private-each topology can run a full episode through `TwoAgentEnv` without
  errors, and the per-agent stats in the result have 3 entries, not 2.
- A 2-agent topology run through the new code gives the SAME success numbers (within seed
  noise) as before your changes — check this against `results/ma_fixed/tb_clamp_s0.json` or
  similar if you have time; if you don't have time, at minimum confirm the existing 2-agent
  tests in `tests/ma/test_env.py` still pass with their existing assertions, since those
  assertions already encode "the 2-agent numbers didn't move".
- Every file you touched has a commit. Commits are small and each says WHY, not WHAT (see
  section 8).

## 8. Commit message format this project uses

Look at `git log --oneline -15` for real examples before writing your first one. Pattern:
first line is `type(scope): short summary of the WHY`, body explains the reasoning and any
correction, ends with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Change that co-author line to whatever identifies you, keep the format.

## 9. If you run out of time or hit something genuinely blocked

Do not leave the tree red. Either:
- Finish the file you're on and commit it green, and stop there, OR
- `git stash` your in-progress edit and leave the tree at the last green commit.

Then write a short section at the bottom of `docs/logs/MA_BUILD_LOG.md`, dated, starting
`**[HANDOVER]**`, saying exactly: what's done, what's left (name the files), what test count
you're at, and any `**[BLOCKED]**` questions from section 3. This is the only way the next
person (human or otherwise) picks this up without re-deriving everything you just learned.
