import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=str, default="diag_runs/single_agent_ippo_avici_s42/eval_trace_temp0.0.json")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/best_ippo_params.pkl")
    args = parser.parse_args()

    if os.path.exists(args.ckpt_path):
        import pickle
        with open(args.ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print("=== Checkpoint Metadata ===")
        for k, v in ckpt.items():
            if not k.endswith("_list"):
                print(f"  {k}: {v}")

    if os.path.exists(args.trace_path):
        with open(args.trace_path, "r") as f:
            trace = json.load(f)

        print("\n=== Trace Metadata ===")
        print(trace.get("metadata"))
        for g in range(8):
            graph_data = trace.get(f"graph_{g}")
            if not graph_data:
                continue
            steps = graph_data["steps"]
            print(f"\n=== Graph {g} ({len(steps)} steps) | Final SHD: {steps[-1]['shd']} ===")
            for i, s in enumerate(steps[:6]):
                act = s["actions"]["agent_0"]
                cat = "INTERVENE" if act.get("cat") == 0 else "NOOP"
                target = act.get("target")
                oracle = s.get("oracle_agreement", {})
                print(f"  Step {i}: Action={cat}(target={target}) | SHD={s['shd']} | Budget={s['budgets']} | OracleScore={oracle.get('score')} ({oracle.get('optimal_action')})")

if __name__ == "__main__":
    main()
