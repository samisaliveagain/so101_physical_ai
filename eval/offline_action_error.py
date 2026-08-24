#!/usr/bin/env python
"""Offline validation for a trained SO-101 policy (Diffusion or VLA-JEPA).

Runs the policy open-loop on the held-out *val* episodes (observations come from the
dataset, i.e. teacher-forced observations) and reports how well predicted actions match
ground truth. This is model-agnostic and the primary gate before any real-robot rollout.

Outputs (under --out-dir):
  report.json                 overall + per-joint MSE/MAE, gripper accuracy
  traj_ep<N>.npz              predicted + ground-truth action arrays per episode
  traj_ep<N>.png              trajectory plot when matplotlib is installed

Usage
-----
  python eval/offline_action_error.py \
      --policy-path $SCRATCH/train/so101_vla_diffusion/checkpoints/last/pretrained_model \
      --repo-id shubham4413/so101_vla \
      --out-dir eval/out/diffusion

  python eval/offline_action_error.py \
      --policy-path $SCRATCH/train/so101_wm_vlajepa/checkpoints/last/pretrained_model \
      --repo-id shubham4413/so101_wm \
      --out-dir eval/out/vlajepa

Notes
-----
* Uses the val episode list written by data/make_splits.py (falls back to all episodes).
* The VLA-JEPA *world-model* loss is tracked during training via --wandb.enable / --eval_freq;
  it is not recomputed here (this script only needs the action outputs, which is all that is
  deployed at inference). Use it to compare checkpoints on action fidelity.
* For a LoRA (PEFT) VLA-JEPA checkpoint, if loading fails, merge adapters into the base first
  (see README "Fallback / PEFT loading").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def load_policy(
    policy_path: str,
    repo_id: str,
    device: str,
    dataset_root: str | None,
    rename_map: dict[str, str],
):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    policy = make_policy(cfg, ds_meta=meta, rename_map=rename_map)
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def val_episodes(repo_id: str) -> list[int] | None:
    f = SPLITS_DIR / (repo_id.replace("/", "__") + ".json")
    if f.exists():
        return json.loads(f.read_text())["val"]
    print("[eval] no split file found; evaluating on ALL episodes")
    return None


def build_observation(item: dict, device: str) -> dict:
    """Turn a dataset item into a single-sample observation batch for select_action."""
    batch = {}
    for k, v in item.items():
        if not k.startswith("observation."):
            continue
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)
    if "task" in item:
        batch["task"] = [item["task"]]
    return batch


@torch.no_grad()
def evaluate(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    episodes: list[int],
    out_dir: Path,
    max_episodes: int,
    device: str,
    rename_map: dict[str, str],
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        print("[eval] matplotlib is unavailable; saving trajectory arrays without PNG plots")

    # LeRobot 0.5.2-dev stores global episode frame ranges in metadata.
    from_idx = dataset.meta.episodes["dataset_from_index"]
    to_idx = dataset.meta.episodes["dataset_to_index"]

    all_pred, all_gt = [], []
    action_names = dataset.meta.features.get("action", {}).get("names") or [
        f"joint_{i}" for i in range(dataset.meta.features["action"]["shape"][0])
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_list = episodes[:max_episodes] if max_episodes else episodes
    for ep in ep_list:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        s, e = int(from_idx[ep]), int(to_idx[ep])
        preds, gts = [], []
        for i in range(s, e):
            item = dataset[i]
            obs = preprocessor(build_observation(item, device))
            action = postprocessor(policy.select_action(obs))  # (1, action_dim), physical units
            preds.append(action.squeeze(0).float().cpu().numpy())
            gts.append(item["action"].float().cpu().numpy())
        preds, gts = np.asarray(preds), np.asarray(gts)
        all_pred.append(preds)
        all_gt.append(gts)
        np.savez_compressed(out_dir / f"traj_ep{ep}.npz", pred=preds, gt=gts)

        # trajectory plot
        if plt is not None:
            n = preds.shape[1]
            fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=True)
            axes = np.atleast_1d(axes)
            for j in range(n):
                axes[j].plot(gts[:, j], label="gt", lw=1.5)
                axes[j].plot(preds[:, j], label="pred", lw=1.2, ls="--")
                axes[j].set_ylabel(action_names[j], fontsize=8)
            axes[0].legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("timestep")
            fig.suptitle(f"episode {ep}  pred vs gt")
            fig.tight_layout()
            fig.savefig(out_dir / f"traj_ep{ep}.png", dpi=110)
            plt.close(fig)

    pred = np.concatenate(all_pred, 0)
    gt = np.concatenate(all_gt, 0)
    err = pred - gt
    per_joint_mse = (err**2).mean(0)
    per_joint_mae = np.abs(err).mean(0)

    # gripper (SO-101 index 5) treated as binary open/close
    gi = 5 if gt.shape[1] > 5 else gt.shape[1] - 1
    gripper_acc = float(((pred[:, gi] > 0.5) == (gt[:, gi] > 0.5)).mean())
    gt_gripper_positive = float((gt[:, gi] > 0.5).mean())
    pred_gripper_positive = float((pred[:, gi] > 0.5).mean())
    gripper_majority_baseline = max(gt_gripper_positive, 1.0 - gt_gripper_positive)

    report = {
        "repo_id": dataset.repo_id,
        "n_episodes": len(ep_list),
        "n_frames": int(gt.shape[0]),
        "action_names": list(action_names),
        "overall_mse": float((err**2).mean()),
        "overall_mae": float(np.abs(err).mean()),
        "per_joint_mse": {action_names[j]: float(per_joint_mse[j]) for j in range(gt.shape[1])},
        "per_joint_mae": {action_names[j]: float(per_joint_mae[j]) for j in range(gt.shape[1])},
        "worst_joint_by_mse": action_names[int(per_joint_mse.argmax())],
        "gripper_binary_accuracy": gripper_acc,
        "gripper_positive_fraction_gt": gt_gripper_positive,
        "gripper_positive_fraction_pred": pred_gripper_positive,
        "gripper_majority_baseline_accuracy": gripper_majority_baseline,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    artifact = "trajectory arrays and plots" if plt is not None else "trajectory arrays"
    print(f"\n[eval] wrote {out_dir/'report.json'} and {len(ep_list)} {artifact}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True, help="path to a saved pretrained_model dir")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--dataset-root",
        default=None,
        help="explicit local LeRobot dataset root (avoids Hub revision lookup)",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--tolerance-s", type=float, default=1e-3)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all val episodes")
    ap.add_argument(
        "--rename-map",
        type=json.loads,
        default={},
        help="JSON object mapping dataset observation keys to policy input keys",
    )
    args = ap.parse_args()
    if not isinstance(args.rename_map, dict):
        ap.error("--rename-map must decode to a JSON object")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    eps = val_episodes(args.repo_id)
    # Load the FULL dataset so episode_data_index[ep] maps to original episode numbers
    # (passing episodes= would re-index the dataset and break that lookup).
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
    )
    episodes = eps if eps is not None else list(range(dataset.meta.total_episodes))
    policy, preprocessor, postprocessor = load_policy(
        args.policy_path, args.repo_id, device, args.dataset_root, args.rename_map
    )
    evaluate(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        episodes,
        args.out_dir,
        args.max_episodes,
        device,
        args.rename_map,
    )


if __name__ == "__main__":
    main()
