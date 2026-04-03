import os, json, tempfile
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import beta as beta_dist

THOR_CAT_SIMPLIFY = {
    "saltshaker": "S/P Shaker", "peppershaker": "S/P Shaker",
    "tomato": "Fruit", "apple": "Fruit",
    "butterknife": "Knife", "boiler": "Kettle",
    "winebottle": "Bottle", "atomizer": "Spray Bottle",
    "remotecontrol": "Remote Control", "soapdispenser": "Soap Dispenser",
    "tissuepaper": "Tissue Paper",
}


def _extract_object_name(obs_scene_bytes):
    try:
        obs = json.loads(obs_scene_bytes.decode("utf-8"))
        raw = obs.get("object_name", "unknown")
        cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
        return cleaned.split()[0] if cleaned else "unknown"
    except Exception:
        return "unknown"


def _simplify(name: str) -> str:
    simp = THOR_CAT_SIMPLIFY.get(name.lower(), name)
    return " ".join(w.capitalize() for w in simp.split())


def _bayesian_ci(successes, total, alpha=0.05):
    if total == 0:
        return 0.0, 0.0
    a, b = 1 + successes, 1 + (total - successes)
    return beta_dist.ppf(alpha / 2, a, b) * 100, beta_dist.ppf(1 - alpha / 2, a, b) * 100


def _copy_group(src, dst):
    for k, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(k, data=item[()])
        elif isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(k))


def _decode_json_sequence(raw_uint8):
    rows = []
    for row in raw_uint8:
        d = json.loads(bytes(row).rstrip(b"\x00").decode("utf-8"))
        flat = []
        for v in d.values():
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        rows.append(flat)
    return np.array(rows, dtype=np.float64)


def _episode_joint_jerk(ep, dt, max_steps=None):
    raw_q = None
    try:
        raw_q = ep["obs"]["agent"]["qpos"][:]
    except KeyError:
        try:
            raw_q = ep["actions"]["joint_pos"][:]
        except KeyError:
            pass
    if raw_q is None:
        return np.nan
    if max_steps is not None:
        raw_q = raw_q[:max_steps]
    q = _decode_json_sequence(raw_q)
    if q.shape[0] < 4:
        return np.nan
    d3 = q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]
    d3 /= dt ** 3
    return float(np.mean(np.linalg.norm(d3, axis=1)))


def _combine_trajectories(folder_path):
    folder = Path(folder_path)
    h5_files = sorted(folder.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 found under {folder_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    out = h5py.File(tmp.name, "w")
    ep = 0
    for src_path in h5_files:
        try:
            src = h5py.File(src_path, "r")
            for tk in [k for k in src.keys() if k.startswith("traj_")]:
                dst = out.create_group(f"episode_{ep:04d}_{tk}")
                _copy_group(src[tk], dst)
                ep += 1
            src.close()
        except Exception as e:
            print(f"Warning: skipping {src_path}: {e}")
    out.close()
    print(f"Combined {ep} episodes from {len(h5_files)} files → {tmp.name}")
    return tmp.name


def eval_to_csv(
    run_path: str,
    policy_name: str,
    reward_threshold: float | None = 0.01,
    output_csv: str = "eval_results.csv",
    dt: float = 0.1,
    number_steps_per_episode: int | None = 450
):

    combined_h5 = _combine_trajectories(run_path)
    per_obj = defaultdict(lambda: {"success": 0, "total": 0, "jerk_joint": []})
    total_s, total_n = 0, 0
    all_jerk_joint = []

    with h5py.File(combined_h5, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("episode_"):
                continue
            ep = f[key]

            if reward_threshold is not None and "rewards" in ep:
                r = ep["rewards"][:]
                if number_steps_per_episode is not None:
                    r = r[:number_steps_per_episode]
                success = r.size > 0 and float(r.max()) >= reward_threshold
            elif "success" in ep:
                s_arr = ep["success"][:]
                if number_steps_per_episode is not None:
                    s_arr = s_arr[:number_steps_per_episode]
                success = bool(max(s_arr)) if len(s_arr) > 0 else False
            else:
                continue

            jj = _episode_joint_jerk(ep, dt, max_steps=number_steps_per_episode)
            obj = _simplify(_extract_object_name(ep["obs_scene"][()])) if "obs_scene" in ep else "Unknown"

            per_obj[obj]["total"] += 1
            per_obj[obj]["success"] += int(success)
            if not np.isnan(jj):
                per_obj[obj]["jerk_joint"].append(jj)
                all_jerk_joint.append(jj)

            total_n += 1
            total_s += int(success)

    # build rows
    rows = []
    for obj in sorted(per_obj):
        d = per_obj[obj]
        s, t = d["success"], d["total"]
        rate = 100.0 * s / t if t else 0.0
        ci_lo, ci_hi = _bayesian_ci(s, t)
        mean_jj = float(np.mean(d["jerk_joint"])) if d["jerk_joint"] else np.nan
        std_jj = float(np.std(d["jerk_joint"])) if d["jerk_joint"] else np.nan
        rows.append(dict(
            policy=policy_name, category=obj, successes=s, total=t,
            success_rate_pct=round(rate, 2),
            ci_95_low_pct=round(ci_lo, 2), ci_95_high_pct=round(ci_hi, 2),
            jerk_joint_mean=round(mean_jj, 6) if not np.isnan(mean_jj) else np.nan,
            jerk_joint_std=round(std_jj, 6) if not np.isnan(std_jj) else np.nan,
        ))

    rate = 100.0 * total_s / total_n if total_n else 0.0
    ci_lo, ci_hi = _bayesian_ci(total_s, total_n)
    mean_jj = float(np.mean(all_jerk_joint)) if all_jerk_joint else np.nan
    std_jj = float(np.std(all_jerk_joint)) if all_jerk_joint else np.nan
    rows.append(dict(
        policy=policy_name, category="OVERALL", successes=total_s, total=total_n,
        success_rate_pct=round(rate, 2),
        ci_95_low_pct=round(ci_lo, 2), ci_95_high_pct=round(ci_hi, 2),
        jerk_joint_mean=round(mean_jj, 6) if not np.isnan(mean_jj) else np.nan,
        jerk_joint_std=round(std_jj, 6) if not np.isnan(std_jj) else np.nan,
    ))

    df = pd.DataFrame(rows)
    with open(output_csv, "w") as fout:
        fout.write(f"# policy_name: {policy_name}\n")
        fout.write(f"# run_path: {run_path}\n")
        fout.write(f"# reward_threshold: {reward_threshold}\n")
        fout.write(f"# dt: {dt}\n")
        fout.write(f"# number_steps_per_episode: {number_steps_per_episode}\n")
        df.to_csv(fout, index=False)

    print(f"\nSaved → {os.path.abspath(output_csv)}")

    os.unlink(combined_h5)

    return df


if __name__ == "__main__":
    RUN_PATH = "/n/fs/robot-data/molmospaces/eval_output/molmo_spaces.evaluation.configs.evaluation_configs:LAPPolicyEvalConfig/20260330_125636"
    # RUN_PATH = "/n/fs/robot-data/molmospaces/eval_output/molmo_spaces.evaluation.configs.evaluation_configs:LAPPolicyEvalConfig/20260330_234853"
    # RUN_PATH = "/n/fs/robot-data/molmospaces/eval_output/molmo_spaces.evaluation.configs.evaluation_configs:LAPPolicyEvalConfig/20260331_091520"

    df = eval_to_csv(
        RUN_PATH,
        policy_name="LAP",
        reward_threshold=0.01,
        output_csv="lap_eval_20260318_142321.csv",
        dt=0.1,
        number_steps_per_episode=300
    )
    print("\nDataFrame preview:")
    print(df)
