#!/usr/bin/env python3
import glob
import sys

import h5py

eval_dir = "/home/irom-lab/projects/molmospaces/eval_output/molmo_spaces.evaluation.configs.evaluation_configs:LAPPolicyEvalConfig/20260312_165709"

h5_files = sorted(glob.glob(f"{eval_dir}/**/*.h5", recursive=True))

total = 0
successes = 0

for h5_path in h5_files:
    try:
        with h5py.File(h5_path, "r") as f:
            for traj_key in f.keys():
                traj = f[traj_key]
                if "success" not in traj:
                    continue
                success = bool(traj["success"][-1])
                total += 1
                if success:
                    successes += 1
    except Exception as e:
        print(f"Error reading {h5_path}: {e}", file=sys.stderr)

if total == 0:
    print("No trajectories found.")
    sys.exit(1)

print(f"Trajectories: {total}")
print(f"Successes:    {successes}")
print(f"Success rate: {successes / total:.1%}")
