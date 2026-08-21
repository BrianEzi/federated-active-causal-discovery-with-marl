import subprocess
import os
import sys

def run_ablation(name, flags):
    print(f"\n=======================================================")
    print(f" Running Ablation: {name}")
    print(f" Flags: {' '.join(flags)}")
    print(f"=======================================================")
    cmd = [sys.executable, "-m", "src.train", "--num_episodes", "10", "--no_save_file"] + flags
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f" SUCCESS: {name}")
        for line in result.stdout.split("\n")[-6:]:
            if line.strip():
                print(f"   {line}")
    else:
        print(f" FAILED: {name}")
        print(result.stderr[-500:])

def main():
    ablations = [
        ("ABL-01a (Soft Shift)", ["--intervention_type", "soft_shift"]),
        ("ABL-01b (Hard Clamping)", ["--intervention_type", "hard"]),
        ("ABL-02 (AVICI Estimator)", ["--estimator_type", "avici"]),
        ("ABL-03 (Sparse Reward)", ["--reward_density", "sparse"]),
        ("ABL-04 (Curiosity Bonus)", ["--intrinsic_coef", "0.05"]),
        ("ABL-05 (Impact Bonus)", ["--impact_coef", "0.10"]),
        ("ABL-06 (No Obs Feedback)", ["--obs_feedback", "false"]),
    ]
    
    for name, flags in ablations:
        run_ablation(name, flags)

if __name__ == "__main__":
    main()
