"""Generate the 12,000-episode re-run of the cells where 4,000 was demonstrably short.

WHY THESE CELLS. `docs/FINDINGS_UNDERTRAINING_2026_09_02.md` shows every competence-floor
exclusion is undertrained at 4,000 episodes, and `FINDINGS_AGENT_COUNT_2026_09_02.md` shows
both reversals in the results chapter are training budget rather than coordination. All of that
is at k=12. k=20 and k=30 already ran at 12,000 in the original sweep, so they are excluded --
re-running them buys nothing. k=4 and k=8 come along because they cost 0.9 core-hours combined
and make the window axis uniform.

Existing 12,000-episode runs are reused rather than repeated: tonight's retrains already cover
four cells completely and seven more at seed 2.

Written in Python, not shell, because four separate jobs tonight were lost to zsh not
word-splitting unquoted variables the way bash does.
"""
from __future__ import annotations
import json, pathlib, shutil, stat

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/sweep/oracle"
DEST = ROOT / "results/sweep12k"
JOBS = DEST / "jobs"
SEEDS = (0, 1, 2)

# k=20 and k=30 already ran at 12,000 episodes in the original sweep.
SKIP = {"k20s50n04b150", "k30s50n04b150"}

def existing_12k(cell: str, seed: int):
    """Any 12,000-episode run of this cell and seed already on disk."""
    for pattern in (f"results/longcheck/{cell}_conv_s{seed}.json",
                    f"results/longcheck/{cell}_long_s{seed}.json",
                    f"results/seeds345/{cell}_s{seed}.json"):
        p = ROOT / pattern
        if p.exists() and json.loads(p.read_text())["config"]["ppo_total_episodes"] == 12000:
            return p
    return None

def main() -> int:
    JOBS.mkdir(parents=True, exist_ok=True)
    cells = sorted({p.stem.rsplit("_s", 1)[0] for p in SWEEP.glob("k*_s*.json")} - SKIP)
    todo, reused = [], []
    for cell in cells:
        cfg = json.loads((SWEEP / f"{cell}_s0.json").read_text())["config"]
        topo = cfg["topology"]
        args = (f"--n_agents {cfg['n_agents']} --private_size {len(topo['private'][0])} "
                f"--n_shared {len(topo['exposed'])} --budget {cfg['budget']}")
        for seed in SEEDS:
            out = DEST / f"{cell}_s{seed}.json"
            have = existing_12k(cell, seed)
            if have is not None:
                shutil.copy2(have, out)
                # The checkpoints must travel with the result. `global_shd_paired.py` loads
                # `<stem>_best.pt` from beside the json and SKIPS the seed with a warning when
                # it is absent, so copying the json alone silently drops reused seeds from the
                # measurement. Three cells were measured on two seeds before this was found.
                # u0500 is the 8,000-episode checkpoint; without it a reused
                # cell silently drops out of the budget comparison.
                for suffix in ("_best.pt", ".pt", "_u0500.pt"):
                    src = have.with_name(have.stem + suffix)
                    if src.exists():
                        shutil.copy2(src, out.with_name(out.stem + suffix))
                reused.append(f"{cell}_s{seed}")
                continue
            if out.exists():
                continue
            job = JOBS / f"{cell}_s{seed}.sh"
            job.write_text(
                "#!/usr/bin/env bash\n"
                "export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
                f"cd {ROOT}\n"
                f'[ -f "{out.relative_to(ROOT)}" ] || .venv/bin/python -u scripts/ma_train.py '
                f"--arm {cell} --seed {seed} {args} "
                "--n_obs 60 --n_int 20 --turn_order round_robin --backend factored "
                "--policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 "
                "--claim_bar 1.0 --reward_criterion claims --per_agent_reward "
                "--episode_mix confounded --normalise_returns --vs_evidence oracle "
                "--train_episodes 12000 --eval_episodes 200 --no_wandb --force "
                "--turn_aware_credit --local_epochs 4 "
                f"--out {out.relative_to(ROOT)}\n")
            job.chmod(job.stat().st_mode | stat.S_IEXEC)
            todo.append(f"{cell}_s{seed}")

    print(f"cells: {len(cells)} (k=20 and k=30 excluded, already at 12,000)")
    print(f"reused from tonight's retrains: {len(reused)}")
    print(f"jobs to run: {len(todo)}")
    est = 0.0
    for name in todo:
        cell = name.rsplit("_s", 1)[0]
        secs = [json.loads(p.read_text())["train_seconds"]
                for p in SWEEP.glob(f"{cell}_s*.json")]
        est += (sum(secs) / len(secs)) * 3 / 2.14      # 4k -> 12k, / measured speedup
    print(f"estimated {est/3600:.1f} core-hours, {est/3600/5:.1f} h at 5 workers")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
