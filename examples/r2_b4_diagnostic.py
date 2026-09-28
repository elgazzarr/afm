"""Diagnostic for Reviewer 2, comment 3 (IJF R2): why does FM's horizon-wise CRPS drop
over the last steps of Fig. B.4 (C-MAPSS FD001)?

Read-only: uses the saved artefacts of the published run, no training or sampling.
    results/realworld/cmapss_FD001_multiseed.json  per-step curves, 3 seeds
    results/realworld/cmapss_FD001_figdata.npz     raw samples, seed 0 only

Background: ``afm.evaluation.metrics.compute_crps`` averages (1{x_(i) >= y} - i/S)^2 over the
sorted samples x_(1..S). With p the fraction of samples below y (the PIT value), this is
approximately (p^3 + (1 - p)^3) / 3. It depends only on where y falls within the predictive
distribution, not on the distribution's width: it is smallest (1/12) when y sits at the
median and largest (1/3) when y lies outside the samples.

Usage: python examples/r2_b4_diagnostic.py [--results-dir results/realworld]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from afm.evaluation.metrics import compute_crps

LAST = 4  # the steps where the FM drop occurs (105-108)


def per_step_width(preds: np.ndarray, level: float = 0.9) -> np.ndarray:
    a = (1 - level) / 2
    lo, hi = np.quantile(preds, a, axis=0), np.quantile(preds, 1 - a, axis=0)
    return (hi - lo).mean(axis=(0, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/realworld")
    args = ap.parse_args()
    root = Path(args.results_dir)
    js = json.loads((root / "cmapss_FD001_multiseed.json").read_text())
    fd = np.load(root / "cmapss_FD001_figdata.npz", allow_pickle=True)
    H = js["H"]
    np.set_printoptions(precision=3, suppress=True, linewidth=160)

    gt = fd["fm_gt"].astype(np.float64)
    assert np.array_equal(fd["afm_gt"], fd["fm_gt"])
    assert not np.isnan(gt).any()
    print(f"[windows] {gt.shape[0]} engines x {gt.shape[2]} sensors at every one of {H} steps; no NaN, no padding")

    print("\n[3 seeds, JSON] per-step score and 90% coverage, seed mean")
    for m in ("FM", "AFM"):
        crps = np.array([s[m]["crps_h"] for s in js["per_seed"]])
        cov = np.array([s[m]["cov90_h"] for s in js["per_seed"]])
        print(f"  {m:3s} score steps {H-7}-{H}: {crps.mean(0)[-8:]}   mean steps 1-100: {crps[:, :100].mean():.3f}")
        print(f"  {m:3s} cov90 steps {H-7}-{H}: {cov.mean(0)[-8:]}   mean steps 1-100: {cov[:, :100].mean():.3f}")
        full, cut = crps.mean(1), crps[:, : H - LAST].mean(1)
        print(f"  {m:3s} aggregate: all steps {full.mean():.4f} +/- {full.std():.4f}; "
              f"without last {LAST}: {cut.mean():.4f} +/- {cut.std():.4f}; per seed {full.round(4)} -> {cut.round(4)}")

    print("\n[seed 0, raw samples] spread, error and position of the observation")
    for m in ("fm", "afm"):
        P = fd[f"{m}_preds"].astype(np.float64)  # (S, B, H, D)
        w = per_step_width(P)
        mae = np.abs(P.mean(0) - gt).mean(axis=(0, 2))
        pit = (P < gt[None]).mean(0)  # (B, H, D)
        central = ((pit > 0.25) & (pit < 0.75)).mean(axis=(0, 2))
        repo = np.array([float(compute_crps(P[:, :, h:h + 1], gt[:, h:h + 1])) for h in range(H)])
        json_repo = np.array(js["per_seed"][0][m.upper()]["crps_h"])
        print(f"  {m.upper():3s} 90% width steps {H-7}-{H}: {w[-8:]}   mean 1-100: {w[:100].mean():.3f}   "
              f"ratio step {H}/mean: {w[-1] / w[:100].mean():.2f}")
        print(f"  {m.upper():3s} MAE of mean steps {H-7}-{H}: {mae[-8:]}   mean 1-100: {mae[:100].mean():.3f}")
        print(f"  {m.upper():3s} share of obs in central 50% steps {H-7}-{H}: {central[-8:]}   "
              f"mean 1-100: {central[:100].mean():.3f} (calibrated: 0.5)")
        print(f"  {m.upper():3s} recomputed score matches JSON seed 0: {np.allclose(repo, json_repo, atol=1e-6)}")
        # Proper CRPS (energy form) for reference only; not reported in the paper.
        t1 = np.abs(P - gt[None]).mean(0)
        t2 = np.abs(P[:, None] - P[None, :]).mean(axis=(0, 1))
        proper = (t1 - 0.5 * t2).mean(axis=(0, 2))
        print(f"  {m.upper():3s} [reference] energy-form CRPS steps {H-7}-{H}: {proper[-8:]}   mean 1-100: {proper[:100].mean():.3f}")


if __name__ == "__main__":
    main()
