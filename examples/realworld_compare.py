"""AFM vs FM head-to-head on a real-world dataset (the paper's core ablation).

Trains both Autoregressive Flow Matching (AFM, single-step) and non-autoregressive
Flow Matching (FM, fixed-window + Brownian prior) on the SAME data and horizon, then
reports CRPS / NRMSE (+ de-normalized RMSE/MAE on the headline channel) for each.

Usage:
  python examples/realworld_compare.py --dataset battery  --holdout B0007 --ctx 66 --epochs 50
  python examples/realworld_compare.py --dataset cmapss   --subset FD001  --ctx 20 --epochs 50
Data: see src/afm/datasets/download.py for file placement under data/.
"""
import argparse
import os
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.battery import load_battery
from afm.datasets.cmapss import load_cmapss
from afm.datasets.eval import evaluate_on_trajectories
from afm.datasets.run_config import build_afm_config
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer, FMTrainer
from afm.models import AutoregressiveFlowMatching, FlowMatching


def _fm_window(slack):
    """A moderate FM training window (<= slack-1, avoiding the Config '30' sentinel)."""
    w = min(40, slack - 1)
    return 28 if w == 30 else w


def build_afm(dim, cfg):
    return AutoregressiveFlowMatching(
        dim_x=dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(0))


def build_fm(dim, cfg):
    # FlowMatching's constructor takes no rnn_hidden (uses dim_h for the RNN).
    return FlowMatching(
        dim_x=dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rngs=nnx.Rngs(1))


def load_problem(args):
    """Returns (ds, cfg, horizon, channel, channel_name, tag)."""
    if args.dataset == "battery":
        ds0 = load_battery(Config(system_name="nasa_battery", system_num_timepoints=80,
                                  context_length=30, fm_prediction_horizon=40,
                                  data_prediction_size=40), holdout=args.holdout)
        L = ds0.meta["raw_length"]; H = L - args.ctx
        fmw = _fm_window(L - args.ctx)                 # FM training window (many start positions)
        cfg = build_afm_config("nasa_battery", length=L, context_length=args.ctx,
                               forecast_horizon=H, fm_prediction_horizon=fmw,
                               flow_nfe=20, flow_n_samples=50)
        ds = load_battery(cfg, holdout=args.holdout)
        return ds, cfg, H, ds.meta["capacity_index"], "capacity (Ah)", f"battery/{args.holdout}"
    else:
        probe = load_cmapss(Config(system_name="cmapss_fd001", system_num_timepoints=128,
                                   context_length=args.ctx, fm_prediction_horizon=40,
                                   data_prediction_size=40), subset=args.subset)
        L = probe.meta["length"]; H = L - args.ctx
        fmw = _fm_window(L - args.ctx)
        cfg = build_afm_config(f"cmapss_{args.subset.lower()}", length=L, context_length=args.ctx,
                               forecast_horizon=H, fm_prediction_horizon=fmw,
                               flow_nfe=20, flow_n_samples=50)
        ds = load_cmapss(cfg, subset=args.subset)
        return ds, cfg, H, 0, ds.meta["sensor_cols"][0], f"cmapss/{args.subset}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["battery", "cmapss"], required=True)
    ap.add_argument("--holdout", default="B0007")       # battery
    ap.add_argument("--subset", default="FD001")        # cmapss
    ap.add_argument("--ctx", type=int, default=66)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fm-epoch-mult", type=int, default=2,
                    help="FM trains for epochs*mult (FM needs more, per the paper).")
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--diagnostics", action="store_true",
                    help="Compute calibration (PI coverage), horizon-wise CRPS/NRMSE, and "
                         "cross-channel correlation; save calibration/horizon plots.")
    args = ap.parse_args()

    ds, cfg, H, ch, chname, tag = load_problem(args)
    (afm_train, afm_test), (fm_train, fm_test) = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj), jnp.asarray(ds.covariates), cfg)

    afm = build_afm(ds.data_dim, cfg)
    AFMTrainer(afm, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=args.epochs, use_wandb=False)

    fm = build_fm(ds.data_dim, cfg)
    FMTrainer(fm, learning_rate=cfg.training_learning_rate, model_name="FM").train(
        fm_train, fm_test, num_epochs=args.epochs * args.fm_epoch_mult, use_wandb=False)

    want_pred = (args.plot_dir is not None) or args.diagnostics
    a = evaluate_on_trajectories(afm, ds, cfg, jr.PRNGKey(42), n_steps_pred=H,
                                 n_samples=cfg.flow_n_samples, nfe=cfg.flow_nfe,
                                 channel=ch, return_predictions=want_pred)
    f = evaluate_on_trajectories(fm, ds, cfg, jr.PRNGKey(43), n_steps_pred=H,
                                 n_samples=cfg.flow_n_samples, nfe=cfg.flow_nfe,
                                 channel=ch, return_predictions=want_pred)
    ares, fres = (a[0], f[0]) if want_pred else (a, f)

    print(f"\n=== AFM vs FM | {tag} | ctx={args.ctx} horizon={H} ===")
    print(f"{'model':<5}{'CRPS':>10}{'NRMSE':>10}{'RMSE_orig':>12}{'MAE_orig':>10}")
    for name, r in (("AFM", ares), ("FM", fres)):
        print(f"{name:<5}{r['crps']:>10.4f}{r['nrmse']:>10.4f}{r['rmse_orig']:>12.4f}{r['mae_orig']:>10.4f}")

    if args.plot_dir is not None:
        from afm.datasets.plots import save_forecast_plot
        os.makedirs(args.plot_dir, exist_ok=True)
        safe = tag.replace("/", "_")
        for name, out in (("AFM", a), ("FM", f)):
            _, preds, gt = out
            path = os.path.join(args.plot_dir, f"{safe}_{name}_{chname.split()[0]}.png")
            save_forecast_plot(ds, preds, gt, cfg, path, channel=ch, channel_name=chname,
                               title=f"{name} forecast — {tag}")
            print(f"  saved plot: {path}")

    if args.diagnostics:
        _run_diagnostics(a, f, tag, args.plot_dir or "results/realworld")


def _run_diagnostics(a, f, tag, out_dir):
    """Calibration (PI coverage), horizon-wise CRPS, and cross-channel correlation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from afm.datasets.diagnostics import summarize

    sa = summarize(a[1], a[2]); sf = summarize(f[1], f[2])
    print(f"\n--- diagnostics | {tag} ---")
    print(f"{'metric':<22}{'AFM':>12}{'FM':>12}{'nominal':>10}")
    print(f"{'coverage@50%':<22}{sa['coverage'][0.5]:>12.3f}{sf['coverage'][0.5]:>12.3f}{0.50:>10.2f}")
    print(f"{'coverage@90%':<22}{sa['coverage'][0.9]:>12.3f}{sf['coverage'][0.9]:>12.3f}{0.90:>10.2f}")
    cda = "n/a" if sa['corr_distance'] is None else f"{sa['corr_distance']:.3f}"
    cdf = "n/a" if sf['corr_distance'] is None else f"{sf['corr_distance']:.3f}"
    print(f"{'corr-matrix dist':<22}{cda:>12}{cdf:>12}{'0 (best)':>10}")

    os.makedirs(out_dir, exist_ok=True)
    safe = tag.replace("/", "_")
    H = len(sa["crps_h"])
    steps = range(1, H + 1)

    # Calibration vs horizon (90% interval): closeness to the nominal 0.9 line is the win.
    plt.close(); plt.figure(figsize=(8, 5))
    plt.plot(steps, sa["coverage_per_h"][0.9], label="AFM")
    plt.plot(steps, sf["coverage_per_h"][0.9], label="FM")
    plt.axhline(0.9, ls="--", color="k", lw=1, label="nominal 0.9")
    plt.xlabel("forecast step"); plt.ylabel("90% PI empirical coverage")
    plt.title(f"Calibration vs horizon — {tag}"); plt.legend(); plt.ylim(0, 1.05)
    p1 = os.path.join(out_dir, f"{safe}_calibration_vs_horizon.png")
    plt.savefig(p1, bbox_inches="tight"); print(f"  saved plot: {p1}")

    # Horizon-wise CRPS.
    plt.close(); plt.figure(figsize=(8, 5))
    plt.plot(steps, sa["crps_h"], label="AFM")
    plt.plot(steps, sf["crps_h"], label="FM")
    plt.xlabel("forecast step"); plt.ylabel("CRPS")
    plt.title(f"Horizon-wise CRPS — {tag}"); plt.legend()
    p2 = os.path.join(out_dir, f"{safe}_crps_vs_horizon.png")
    plt.savefig(p2, bbox_inches="tight"); print(f"  saved plot: {p2}")


if __name__ == "__main__":
    main()
