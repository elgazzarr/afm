"""Generate publication-quality appendix figures (PDF) and LaTeX tables from the
multi-seed run artifacts produced by realworld_multiseed.py:
  results/realworld/<safe>_multiseed.json   (per-seed scalars + horizon curves)
  results/realworld/<safe>_figdata.npz      (seed-0 preds/gt/history + scaler)

Outputs (default into results/realworld/appendix/):
  figures (PDF): forecast fans (AFM|FM) per dataset, calibration-vs-horizon,
                 CRPS-vs-horizon, C-MAPSS cross-sensor correlation heatmaps.
  tables (.tex): multi-seed AFM-vs-FM, battery-vs-published, C-MAPSS diagnostics.

Usage:  python examples/make_appendix.py --in-dir results/realworld
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
AFM_C, FM_C = "#1f77b4", "#d62728"
RATED_AH = 2.0  # Li-ion rated capacity, for normalized-SOH conversion

# (tag, safe, headline channel label, x-axis label)
DATASETS = [
    ("battery/B0007", "battery_B0007", "capacity (Ah)", "discharge cycle"),
    ("cmapss/FD001", "cmapss_FD001", "sensor (normalized)", "engine cycle"),
]


def _load(in_dir, safe):
    jp = os.path.join(in_dir, f"{safe}_multiseed.json")
    npz = os.path.join(in_dir, f"{safe}_figdata.npz")
    if not (os.path.exists(jp) and os.path.exists(npz)):
        return None, None
    with open(jp) as fh:
        js = json.load(fh)
    return js, dict(np.load(npz, allow_pickle=True))


def _denorm(arr, mean, std, ch):
    return arr * (std[ch] + 1e-8) + mean[ch]


# ---------- figures ----------
def fig_forecast_fan(out, tag, safe, chlabel, xlabel, fd):
    ch = int(fd["channel"]); ctx = int(fd["ctx"]); H = int(fd["H"])
    mean, std = fd["scaler_mean"], fd["scaler_std"]
    hist = _denorm(fd["history"][0, :, ch], mean, std, ch)            # (ctx,)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, name, key in ((axes[0], "AFM", "afm"), (axes[1], "FM", "fm")):
        preds = _denorm(fd[f"{key}_preds"][:, 0, :, ch], mean, std, ch)   # (S,H)
        gt = _denorm(fd[f"{key}_gt"][0, :, ch], mean, std, ch)            # (H,)
        xt_h = np.arange(ctx); xt_f = np.arange(ctx - 1, ctx + H)
        med = np.median(preds, 0)
        lo90, hi90 = np.percentile(preds, [5, 95], 0)
        lo50, hi50 = np.percentile(preds, [25, 75], 0)
        link = lambda a: np.concatenate([hist[-1:], a])
        ax.axvspan(0, ctx - 1, color="0.92", lw=0)
        ax.fill_between(xt_f, link(lo90), link(hi90), color=AFM_C if name == "AFM" else FM_C,
                        alpha=0.18, lw=0, label="90% PI")
        ax.fill_between(xt_f, link(lo50), link(hi50), color=AFM_C if name == "AFM" else FM_C,
                        alpha=0.30, lw=0, label="50% PI")
        ax.plot(xt_h, hist, color="0.35", lw=1.6)
        ax.plot(xt_f, link(gt), color="k", lw=1.8, label="ground truth")
        ax.plot(xt_f, link(med), color=AFM_C if name == "AFM" else FM_C, lw=1.8, ls="--", label="median")
        ax.set_title(f"{name}"); ax.set_xlabel(xlabel)
        ax.legend(loc="best", framealpha=0.9)
    axes[0].set_ylabel(chlabel)
    fig.suptitle(f"Probabilistic forecast — {tag} (context shaded)", y=1.02)
    p = os.path.join(out, f"{safe}_forecast_fan.pdf"); fig.savefig(p); plt.close(fig)
    return p


def _seed_curves(js, model, field):
    arr = np.array([r[model][field] for r in js["per_seed"]])  # (seeds,H)
    return arr.mean(0), arr.std(0)


def fig_calibration(out, tag, safe, js):
    H = js["H"]; steps = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, c in (("AFM", AFM_C), ("FM", FM_C)):
        mu, sd = _seed_curves(js, model, "cov90_h")
        ax.plot(steps, mu, color=c, lw=1.8, label=model)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    ax.axhline(0.9, ls="--", color="k", lw=1, label="nominal (0.90)")
    ax.set_xlabel("forecast step"); ax.set_ylabel("90% PI empirical coverage")
    ax.set_ylim(0, 1.05); ax.set_title(f"Calibration vs. horizon — {tag}")
    ax.legend(loc="best")
    p = os.path.join(out, f"{safe}_calibration_vs_horizon.pdf"); fig.savefig(p); plt.close(fig)
    return p


def fig_crps(out, tag, safe, js):
    H = js["H"]; steps = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, c in (("AFM", AFM_C), ("FM", FM_C)):
        mu, sd = _seed_curves(js, model, "crps_h")
        ax.plot(steps, mu, color=c, lw=1.8, label=model)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("forecast step"); ax.set_ylabel("CRPS")
    ax.set_title(f"Horizon-wise CRPS — {tag}"); ax.legend(loc="best")
    p = os.path.join(out, f"{safe}_crps_vs_horizon.pdf"); fig.savefig(p); plt.close(fig)
    return p


def fig_correlation(out, tag, safe, fd):
    """3-panel cross-channel correlation heatmaps (truth / AFM / FM)."""
    dim = fd["afm_gt"].shape[-1]
    if dim < 2:
        return None
    true_c = np.corrcoef(fd["afm_gt"].reshape(-1, dim).T)
    afm_c = np.corrcoef(fd["afm_preds"].reshape(-1, dim).T)
    fm_c = np.corrcoef(fd["fm_preds"].reshape(-1, dim).T)
    da = np.linalg.norm(true_c - afm_c); df = np.linalg.norm(true_c - fm_c)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, M, ttl in ((axes[0], true_c, "Ground truth"),
                       (axes[1], afm_c, f"AFM  (dist {da:.2f})"),
                       (axes[2], fm_c, f"FM  (dist {df:.2f})")):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="correlation")
    fig.suptitle(f"Cross-sensor correlation structure — {tag} (lower distance = better)", y=1.03)
    p = os.path.join(out, f"{safe}_correlation.pdf"); fig.savefig(p); plt.close(fig)
    return p


# ---------- tables ----------
def _pm(ms):
    m, s = ms
    return "--" if m is None else f"${m:.3f}\\!\\pm\\!{s:.3f}$"


def tbl_multiseed(out, data):
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Dataset & Model & CRPS & NRMSE & RMSE & cov$_{50}$ & cov$_{90}$ & corr-dist \\",
             r"\midrule"]
    for tag, safe, *_ in DATASETS:
        js = data.get(safe)
        if not js:
            continue
        for model in ("AFM", "FM"):
            a = js["aggregate"][model]
            name = tag.split("/")[1]
            row = (f"{name if model=='AFM' else ''} & {model} & "
                   f"{_pm(a['crps'])} & {_pm(a['nrmse'])} & {_pm(a['rmse_orig'])} & "
                   f"{_pm(a['cov50'])} & {_pm(a['cov90'])} & {_pm(a['corr_dist'])} \\\\")
            lines.append(row)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    p = os.path.join(out, "table_multiseed.tex"); open(p, "w").write("\n".join(lines))
    return p


def tbl_battery_published(out, data):
    js = data.get("battery_B0007")
    if not js:
        return None
    a = js["aggregate"]
    afm_ah, afm_sd = a["AFM"]["rmse_orig"]; fm_ah, fm_sd = a["FM"]["rmse_orig"]
    rows = [
        ("AFM (ours)", f"${afm_ah:.3f}\\!\\pm\\!{afm_sd:.3f}$", f"{afm_ah/RATED_AH:.3f}",
         "forecast from ctx, leave-one-battery-out"),
        ("FM (ours)", f"${fm_ah:.3f}\\!\\pm\\!{fm_sd:.3f}$", f"{fm_ah/RATED_AH:.3f}",
         "forecast from ctx, leave-one-battery-out"),
        ("GPR (Richardson et al., 2017)", "--", "$\\approx$0.06--0.08",
         "forecast future capacity, $\\sim$40-cycle horizon"),
    ]
    lines = [r"\begin{tabular}{lccl}", r"\toprule",
             r"Method & capacity RMSE (Ah) & norm.\ SOH RMSE & Protocol \\", r"\midrule"]
    lines += [f"{m} & {ah} & {nm} & {pr} \\\\" for m, ah, nm, pr in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(out, "table_battery_published.tex"); open(p, "w").write("\n".join(lines))
    return p


def tbl_cmapss_diag(out, data):
    js = data.get("cmapss_FD001")
    if not js:
        return None
    a = js["aggregate"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Metric & AFM & FM & nominal/best \\", r"\midrule",
             f"coverage@50\\% & {_pm(a['AFM']['cov50'])} & {_pm(a['FM']['cov50'])} & 0.50 \\\\",
             f"coverage@90\\% & {_pm(a['AFM']['cov90'])} & {_pm(a['FM']['cov90'])} & 0.90 \\\\",
             f"cross-sensor corr-dist & {_pm(a['AFM']['corr_dist'])} & {_pm(a['FM']['corr_dist'])} & 0 \\\\",
             r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(out, "table_cmapss_diagnostics.tex"); open(p, "w").write("\n".join(lines))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/realworld")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.in_dir, "appendix")
    os.makedirs(out, exist_ok=True)

    data, fdata = {}, {}
    for tag, safe, chlabel, xlabel in DATASETS:
        js, fd = _load(args.in_dir, safe)
        if js is None:
            print(f"[skip] missing artifacts for {safe}")
            continue
        data[safe] = js; fdata[safe] = fd
        print("FIG", fig_forecast_fan(out, tag, safe, chlabel, xlabel, fd))
        print("FIG", fig_calibration(out, tag, safe, js))
        print("FIG", fig_crps(out, tag, safe, js))
        c = fig_correlation(out, tag, safe, fd)
        if c:
            print("FIG", c)
    for fn in (tbl_multiseed, tbl_battery_published, tbl_cmapss_diag):
        p = fn(out, data)
        if p:
            print("TAB", p)
    print(f"\nAppendix artifacts in: {out}")


if __name__ == "__main__":
    main()
