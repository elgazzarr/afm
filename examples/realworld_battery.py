# examples/realworld_battery.py
"""Capacity-fade forecasting on the NASA Li-ion Battery Aging dataset (leave-one-battery-out).

Usage:  python examples/realworld_battery.py --epochs 100 --holdout B0018
Data:   place B0005.mat B0006.mat B0007.mat B0018.mat under data/battery/ (see download.py).
"""
import argparse
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.battery import load_battery
from afm.datasets.eval import evaluate_on_trajectories
from afm.datasets.run_config import build_afm_config
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer
from afm.models import AutoregressiveFlowMatching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--holdout", default="B0018")
    ap.add_argument("--ctx", type=int, default=30)
    ap.add_argument("--univariate", action="store_true")
    ap.add_argument("--plot-dir", default=None,
                    help="If set, save a capacity forecast-sample plot here.")
    args = ap.parse_args()

    # Probe pass to determine the actual trajectory length from the data files.
    ds0 = load_battery(Config(system_name="nasa_battery", system_num_timepoints=80,
                              context_length=30, fm_prediction_horizon=40,
                              data_prediction_size=40),
                       holdout=args.holdout, univariate=args.univariate)
    L = ds0.meta["raw_length"]
    pred = L - args.ctx                       # forecast everything after context

    cfg = build_afm_config("nasa_battery", length=L, context_length=args.ctx,
                           forecast_horizon=pred, data_batch_size=64,
                           steps_per_epoch=100, flow_nfe=20, flow_n_samples=50)
    ds = load_battery(cfg, holdout=args.holdout, univariate=args.univariate)

    (afm_train, afm_test), _ = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj),
        jnp.asarray(ds.covariates), cfg)

    model = AutoregressiveFlowMatching(
        dim_x=ds.data_dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(0))
    AFMTrainer(model, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=args.epochs, use_wandb=False)

    cap_idx = ds.meta["capacity_index"]
    out = evaluate_on_trajectories(model, ds, cfg, jr.PRNGKey(42),
                                   n_steps_pred=pred, n_samples=cfg.flow_n_samples,
                                   nfe=cfg.flow_nfe, channel=cap_idx,
                                   return_predictions=args.plot_dir is not None)
    res = out[0] if args.plot_dir is not None else out
    print(f"[battery holdout={args.holdout}] "
          f"CRPS={res['crps']:.4f} NRMSE={res['nrmse']:.4f} "
          f"capacity_RMSE={res['rmse_orig']:.4f} Ah  MAE={res['mae_orig']:.4f} Ah")

    if args.plot_dir is not None:
        import os
        from afm.datasets.plots import save_forecast_plot
        _, preds, gt = out
        os.makedirs(args.plot_dir, exist_ok=True)
        path = os.path.join(args.plot_dir, f"battery_{args.holdout}_capacity.png")
        save_forecast_plot(ds, preds, gt, cfg, path, channel=cap_idx,
                           channel_name="capacity (Ah)",
                           title=f"AFM capacity-fade forecast — holdout {args.holdout}")
        print(f"  saved plot: {path}")


if __name__ == "__main__":
    main()
