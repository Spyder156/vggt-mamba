"""Minimal VGGT test, matching the README quick-start EXACTLY.

The README pattern:
    images = load_and_preprocess_images(image_names).to(device)
    predictions = model(images)

i.e. NO unsqueeze (4D input, model adds batch dim internally), NO custom
autocast wrapping. Just call the model the documented way.

We feed the same 24 fr2/desk frames as the previous test. Report the RAW
VGGT outputs (no Umeyama, no scale alignment). Compare to TUM GT directly.

The question this answers: if I use VGGT the textbook way and don't touch the
output, is the result already metric (because joint global solve resolves
scale)? Or is it still up to similarity (the scale ambiguity is fundamental)?

  Expected outcomes:
    - If raw translations match TUM GT magnitudes (within 5-10%):
        joint global solve gives metric → my unsqueeze or earlier prep was the bug
    - If raw translations are ~0.44× of GT magnitudes (consistent ratio):
        scale ambiguity is fundamental — VGGT predicts up to similarity
        regardless of how many frames you give it
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/vggt")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt.models.vggt import VGGT                                                    # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images                            # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri                         # noqa: E402

from vggt_mamba.data.tum_rgbd import sync_sequence                                   # noqa: E402

SEQ = "rgbd_dataset_freiburg2_desk"
DATA_ROOT = Path("/workspace/datasets/tum_rgbd")
N_FRAMES = 24
CKPT_PATH = Path("/workspace/datasets/weights/vggt/model.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # === Pick 24 fr2/desk frames, sampled evenly ===
    recs_all = sync_sequence(DATA_ROOT / SEQ)
    idx = np.linspace(0, len(recs_all) - 1, N_FRAMES).astype(int).tolist()
    recs = [recs_all[i] for i in idx]
    image_names = [str(r.rgb_path) for r in recs]
    print(f"Sequence: {SEQ}")
    print(f"Total frames: {len(recs_all)}; sampled: {len(recs)}")

    # === README quick-start (verbatim shape) ===
    print(f"Loading VGGT from {CKPT_PATH} ...")
    model = VGGT().to(device)
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    images = load_and_preprocess_images(image_names).to(device)                       # (S, 3, H, W)
    print(f"images shape (4D, no unsqueeze): {tuple(images.shape)}")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)                                               # NO unsqueeze
    print(f"output keys: {list(predictions.keys())}")
    print(f"pose_enc shape: {tuple(predictions['pose_enc'].shape)}")                  # → (1, S, 9)
    print(f"depth shape:    {tuple(predictions['depth'].shape)}")                     # → (1, S, H, W, 1)
    print(f"images shape:   {tuple(predictions['images'].shape)}")                    # → (1, S, 3, H, W)

    # Convert pose_enc → extrinsics (cam-from-world)
    img_h, img_w = images.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(predictions["pose_enc"], (img_h, img_w))
    extr_np = extr[0].cpu().float().numpy()                                           # (S, 3, 4)

    # World-from-cam by inverting (cam-from-world)
    pred_wc = np.zeros((N_FRAMES, 4, 4))
    pred_wc[:, 3, 3] = 1.0
    for i in range(N_FRAMES):
        R_cw = extr_np[i, :3, :3]
        t_cw = extr_np[i, :3, 3]
        pred_wc[i, :3, :3] = R_cw.T
        pred_wc[i, :3, 3] = -R_cw.T @ t_cw

    # Relativize to first frame so we can compare to GT
    P0_inv = np.linalg.inv(pred_wc[0])
    pred_rel = np.einsum("ij,njk->nik", P0_inv, pred_wc)

    # GT relative to first frame, same way as the comparison script
    gt_abs = np.stack([r.pose_w_c for r in recs])
    P0_gt_inv = np.linalg.inv(gt_abs[0])
    gt_rel = np.einsum("ij,njk->nik", P0_gt_inv, gt_abs)

    print(f"\n=== RAW VGGT vs GT, NO Umeyama, NO scale correction ===")
    print(f"  {'i':>3}  {'frame':>5}  {'|t|_pred':>10}  {'|t|_gt':>10}  {'ratio':>8}")
    ratios = []
    for i, fi in enumerate(idx):
        t_p = pred_rel[i, :3, 3]
        t_g = gt_rel[i, :3, 3]
        nrm_p = float(np.linalg.norm(t_p))
        nrm_g = float(np.linalg.norm(t_g))
        ratio = nrm_p / nrm_g if nrm_g > 1e-6 else float("nan")
        if i > 0:
            ratios.append(ratio)
        print(f"  {i:>3}  {fi:>5}  {nrm_p:>10.3f}  {nrm_g:>10.3f}  {ratio:>8.3f}")
    if ratios:
        ratios_a = np.array(ratios)
        print(f"\n  Ratio statistics (excluding frame 0 which is identity by construction):")
        print(f"    mean   = {ratios_a.mean():.4f}")
        print(f"    median = {float(np.median(ratios_a)):.4f}")
        print(f"    std    = {ratios_a.std():.4f}")
        print(f"    min    = {ratios_a.min():.4f}")
        print(f"    max    = {ratios_a.max():.4f}")
        print(f"\n  If ratios are tightly clustered (low std), VGGT picked one global scale factor")
        print(f"  consistently across all frames → scale ambiguity is fundamental, not per-frame noise.")
        print(f"  If ratios vary wildly, the joint solve didn't really constrain scale across frames.")


if __name__ == "__main__":
    main()
