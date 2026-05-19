"""Plot Phase 1 training curves from a metrics.jsonl log.

Usage:
    ./docker/run.sh python viz/plot_phase1_curves.py \\
        --log experiments/phase1_tokenizer_probe/dinov2/metrics.jsonl

    # Or overlay multiple runs:
    ./docker/run.sh python viz/plot_phase1_curves.py \\
        --log experiments/phase1_tokenizer_probe/dinov2/metrics.jsonl \\
        --log experiments/phase1_tokenizer_probe/vjepa/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True, type=Path,
                   help="path to metrics.jsonl (repeatable)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "output/phase1_curves.png")
    return p.parse_args()


def load_log(path: Path) -> dict[str, list]:
    train = {"step": [], "loss_total": [], "loss_l1": [], "loss_log": [], "loss_mvc": []}
    eval_ = {"step": [], "mvc_mean": [], "mvc_min": [], "mvc_max": []}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "eval":
                eval_["step"].append(rec["step"])
                eval_["mvc_mean"].append(rec.get("mvc_mean"))
                eval_["mvc_min"].append(rec.get("mvc_min"))
                eval_["mvc_max"].append(rec.get("mvc_max"))
            else:
                for k in train:
                    if k in rec:
                        train[k].append(rec[k])
    return {"train": train, "eval": eval_, "label": path.parent.name}


def main() -> None:
    args = parse_args()
    runs = [load_log(p) for p in args.log]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for run in runs:
        tr = run["train"]
        ev = run["eval"]
        lbl = run["label"]

        axes[0, 0].plot(tr["step"], tr["loss_total"], label=lbl)
        axes[0, 1].plot(tr["step"], tr["loss_l1"], label=lbl)
        axes[1, 0].plot(tr["step"], tr["loss_mvc"], label=f"{lbl} train")
        if ev["step"]:
            axes[1, 0].plot(ev["step"], ev["mvc_mean"], "o-", label=f"{lbl} eval", linewidth=2)
        axes[1, 1].plot(tr["step"], tr["loss_log"], label=lbl)

    axes[0, 0].set_title("total loss"); axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("L1 pointmap loss"); axes[0, 1].set_yscale("log")
    axes[1, 0].set_title("multi-view consistency  (train vs eval)")
    axes[1, 1].set_title("scale-invariant log depth"); axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[viz/curves] saved {args.out}")
    for run in runs:
        ev = run["eval"]
        if ev["step"]:
            print(f"[viz/curves] {run['label']}: final eval mvc_mean={ev['mvc_mean'][-1]:.4f} "
                  f"at step {ev['step'][-1]}")


if __name__ == "__main__":
    main()
