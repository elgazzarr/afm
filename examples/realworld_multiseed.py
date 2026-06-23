"""Multi-seed AFM-vs-FM robustness on a real-world dataset.

Runs N seeds, reports mean +/- std for CRPS / NRMSE / RMSE / MAE and the
diagnostics (PI coverage @50/90, cross-channel correlation distance), and saves
seed-averaged calibration-vs-horizon and CRPS-vs-horizon plots (mean +/- std band).
Self-contained (does not import from other example scripts).

Usage:
  python examples/realworld_multiseed.py --dataset cmapss  --subset FD001 --ctx 20 --seeds 3 --epochs 50
  python examples/realworld_multiseed.py --dataset battery --holdout B0007 --ctx 66 --seeds 3 --epochs 50
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.battery import load_battery
from afm.datasets.cmapss import load_cmapss
from afm.datasets.eval import evaluate_on_trajectories
from afm.datasets.diagnostics import summarize
from afm.datasets.run_config import build_afm_config
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer, FMTrainer
from afm.models import AutoregressiveFlowMatching, FlowMatching


def _fm_window(slack):
    w = min(40, slack - 1)
    return 28 if w == 30 else w


def load_problem(args):
    if args.dataset == "battery":
        ds0 = load_battery(Config(system_name="nasa_battery", system_num_timepoints=80,
                                  context_length=30, fm_prediction_horizon=40,
                                  data_prediction_size=40), holdout=args.holdout)
        L = ds0.meta["raw_length"]; H = L - args.ctx
        cfg = build_afm_config("nasa_battery", length=L, context_length=args.ctx,
                               forecast_horizon=H, fm_prediction_horizon=_fm_window(L - args.ctx),
                               flow_nfe=20, flow_n_samples=50)
        ds = load_battery(cfg, holdout=args.holdout)
        return ds, cfg, H, ds.meta["capacity_index"], f"battery/{args.holdout}"
    probe = load_cmapss(Config(system_name="cmapss_fd001", system_num_timepoints=128,
                               context_length=args.ctx, fm_prediction_horizon=40,
                               data_prediction_size=40), subset=args.subset)
    L = probe.meta["length"]; H = L - args.ctx
    cfg = build_afm_config(f"cmapss_{args.subset.lower()}", length=L, context_length=args.ctx,
                           forecast_horizon=H, fm_prediction_horizon=_fm_window(L - args.ctx),
                           flow_nfe=20, flow_n_samples=50)
    ds = load_cmapss(cfg, subset=args.subset)
    return ds, cfg, H, 0, f"cmapss/{args.subset}"


def run_seed(ds, cfg, H, ch, epochs, fm_mult, seed):
    """Train AFM + FM for one seed; return (afm_record, fm_record) dicts."""
    (afm_train, afm_test), (fm_train, fm_test) = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj), jnp.asarray(ds.covariates), cfg)

    afm = AutoregressiveFlowMatching(
        dim_x=ds.data_dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(seed))
    AFMTrainer(afm, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=epochs, seed=seed, use_wandb=False)

    fm = FlowMatching(
        dim_x=ds.data_dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rngs=nnx.Rngs(seed + 100))
    FMTrainer(fm, learning_rate=cfg.training_learning_rate, model_name="FM").train(
        fm_train, fm_test, num_epochs=epochs * fm_mult, seed=seed + 100, use_wandb=False)

    recs = {}
    models = {"AFM": afm, "FM": fm}
    preds_store = {}
    for name, model, key in (("AFM", afm, 1000 + seed), ("FM", fm, 2000 + seed)):
        m, preds, gt = evaluate_on_trajectories(
            model, ds, cfg, jr.PRNGKey(key), n_steps_pred=H,
            n_samples=cfg.flow_n_samples, nfe=cfg.flow_nfe, channel=ch, return_predictions=True)
        s = summarize(preds, gt)
        recs[name] = {
            "crps": m["crps"], "nrmse": m["nrmse"], "rmse_orig": m["rmse_orig"], "mae_orig": m["mae_orig"],
            "cov50": s["coverage"][0.5], "cov90": s["coverage"][0.9],
            "corr_dist": s["corr_distance"],
            "crps_h": s["crps_h"].tolist(), "cov90_h": s["coverage_per_h"][0.9].tolist(),
            "cov50_h": s["coverage_per_h"][0.5].tolist(),
        }
        preds_store[name] = (np.asarray(preds), np.asarray(gt))
    return recs, models, preds_store


def _ms(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["battery", "cmapss"], required=True)
    ap.add_argument("--holdout", default="B0007")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--ctx", type=int, default=66)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fm-epoch-mult", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", default="results/realworld")
    args = ap.parse_args()

    import pickle
    ds, cfg, H, ch, tag = load_problem(args)
    safe = tag.replace("/", "_")
    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = os.path.join(args.out_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    per_seed = []
    seed0_preds = None
    for s in range(args.seeds):
        print(f"[multiseed {tag}] seed {s+1}/{args.seeds} ...", flush=True)
        recs, models, preds_store = run_seed(ds, cfg, H, ch, args.epochs, args.fm_epoch_mult, s)
        per_seed.append(recs)
        # Save trained model states (nnx state pytree pickles cleanly) for reproducibility.
        for name, model in models.items():
            with open(os.path.join(models_dir, f"{safe}_{name}_seed{s}.pkl"), "wb") as fh:
                pickle.dump(nnx.state(model), fh)
        if s == 0:
            seed0_preds = preds_store

    # Figure-data bundle (seed-0 predictions + de-norm stats) so appendix figures are
    # reproducible without retraining.
    sc = ds.meta["scaler"]
    sensor_cols = ds.meta.get("sensor_cols", [])
    np.savez(
        os.path.join(args.out_dir, f"{safe}_figdata.npz"),
        afm_preds=seed0_preds["AFM"][0], afm_gt=seed0_preds["AFM"][1],
        fm_preds=seed0_preds["FM"][0], fm_gt=seed0_preds["FM"][1],
        history=np.asarray(ds.test_traj)[:, :int(args.ctx)],
        scaler_mean=np.asarray(sc.mean), scaler_std=np.asarray(sc.std),
        channel=int(ch), ctx=int(args.ctx), H=int(H),
        tag=np.array(tag), sensor_cols=np.array(sensor_cols, dtype=object),
    )

    scalars = ("crps", "nrmse", "rmse_orig", "mae_orig", "cov50", "cov90", "corr_dist")
    agg = {}
    for model in ("AFM", "FM"):
        agg[model] = {k: _ms([r[model][k] for r in per_seed]) for k in scalars}

    print(f"\n=== multi-seed AFM vs FM | {tag} | ctx={args.ctx} H={H} | seeds={args.seeds} ===")
    print(f"{'metric':<12}{'AFM (mean+/-std)':>22}{'FM (mean+/-std)':>22}")
    for k in scalars:
        am, asd = agg["AFM"][k]; fm_, fsd = agg["FM"][k]
        af = "n/a" if am is None else f"{am:.4f}+/-{asd:.4f}"
        ff = "n/a" if fm_ is None else f"{fm_:.4f}+/-{fsd:.4f}"
        print(f"{k:<12}{af:>22}{ff:>22}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(model, field):
        arr = np.array([r[model][field] for r in per_seed])  # (seeds, H)
        return arr.mean(0), arr.std(0)

    steps = np.arange(1, H + 1)
    # Calibration vs horizon (90%), seed-averaged with +/-std band.
    plt.close(); plt.figure(figsize=(8, 5))
    for model, c in (("AFM", "C0"), ("FM", "C1")):
        mu, sd = curve(model, "cov90_h")
        plt.plot(steps, mu, color=c, label=model)
        plt.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.2)
    plt.axhline(0.9, ls="--", color="k", lw=1, label="nominal 0.9")
    plt.xlabel("forecast step"); plt.ylabel("90% PI empirical coverage")
    plt.title(f"Calibration vs horizon ({args.seeds} seeds) — {tag}")
    plt.legend(); plt.ylim(0, 1.05)
    p1 = os.path.join(args.out_dir, f"{safe}_calibration_vs_horizon_ms.png")
    plt.savefig(p1, bbox_inches="tight"); print(f"saved {p1}")

    plt.close(); plt.figure(figsize=(8, 5))
    for model, c in (("AFM", "C0"), ("FM", "C1")):
        mu, sd = curve(model, "crps_h")
        plt.plot(steps, mu, color=c, label=model)
        plt.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.2)
    plt.xlabel("forecast step"); plt.ylabel("CRPS")
    plt.title(f"Horizon-wise CRPS ({args.seeds} seeds) — {tag}")
    plt.legend()
    p2 = os.path.join(args.out_dir, f"{safe}_crps_vs_horizon_ms.png")
    plt.savefig(p2, bbox_inches="tight"); print(f"saved {p2}")

    out_json = os.path.join(args.out_dir, f"{safe}_multiseed.json")
    with open(out_json, "w") as fh:
        json.dump({"tag": tag, "ctx": args.ctx, "H": H, "seeds": args.seeds,
                   "aggregate": agg, "per_seed": per_seed}, fh, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
