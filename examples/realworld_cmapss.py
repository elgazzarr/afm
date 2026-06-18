"""C-MAPSS turbofan: (A) multivariate sensor-trajectory forecasting (CRPS/NRMSE),
and (B) RUL prediction via forecasting a 1-D health index to its failure threshold
(RUL-RMSE + NASA Score).  FD001 by default.

Usage:  python examples/realworld_cmapss.py --epochs 100 --subset FD001
Data:   train_FD001.txt test_FD001.txt RUL_FD001.txt under data/cmapss/ (see download.py).
"""
import argparse
import numpy as np
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.cmapss import load_cmapss, load_cmapss_rul, derive_rul, nasa_score
from afm.datasets.eval import evaluate_on_trajectories
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer
from afm.models import AutoregressiveFlowMatching

def make_model(dim, cfg):
    return AutoregressiveFlowMatching(
        dim_x=dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument("--cap", type=int, default=125)
    args = ap.parse_args()

    # ---------- (A) multivariate forecasting ----------
    probe = load_cmapss(Config(system_name="cmapss_fd001", system_num_timepoints=128,
                               context_length=args.ctx, fm_prediction_horizon=40,
                               data_prediction_size=40), subset=args.subset)
    L = probe.meta["length"]; pred = L - args.ctx
    cfg = Config(system_name=f"cmapss_{args.subset.lower()}", system_num_timepoints=L,
                 context_length=args.ctx, afm_prediction_horizon=1,
                 fm_prediction_horizon=pred, data_prediction_size=pred,
                 data_batch_size=64, steps_per_epoch=100, model_covariates_dim=1,
                 flow_nfe=20, flow_n_samples=50)
    ds = load_cmapss(cfg, subset=args.subset)
    (afm_train, afm_test), _ = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj), jnp.asarray(ds.covariates), cfg)
    fmodel = make_model(ds.data_dim, cfg)
    AFMTrainer(fmodel, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=args.epochs, use_wandb=False)
    fres = evaluate_on_trajectories(fmodel, ds, cfg, jr.PRNGKey(1),
                                    n_steps_pred=pred, n_samples=cfg.flow_n_samples, nfe=cfg.flow_nfe)
    print(f"[cmapss {args.subset} forecasting] CRPS={fres['crps']:.4f} NRMSE={fres['nrmse']:.4f}")

    # ---------- (B) RUL via health-index forecasting ----------
    rul = load_cmapss_rul(cfg, subset=args.subset)
    hicfg = Config(system_name=f"cmapss_{args.subset.lower()}",
                   system_num_timepoints=rul["train_hi"].shape[1],
                   context_length=args.ctx, afm_prediction_horizon=1,
                   fm_prediction_horizon=rul["train_hi"].shape[1] - args.ctx,
                   data_prediction_size=rul["train_hi"].shape[1] - args.ctx,
                   data_batch_size=64, steps_per_epoch=100, model_covariates_dim=1,
                   flow_nfe=20, flow_n_samples=50)
    cov = np.linspace(0, 1, rul["train_hi"].shape[1]).astype(np.float32)
    (hi_train, hi_test), _ = create_infinite_dataloaders(
        jnp.asarray(rul["train_hi"]), jnp.asarray(rul["train_hi"][:1]), jnp.asarray(cov), hicfg)
    himodel = make_model(1, hicfg)
    AFMTrainer(himodel, learning_rate=hicfg.training_learning_rate, model_name="AFM-HI").train(
        hi_train, hi_test, num_epochs=args.epochs, use_wandb=False)

    horizon = args.cap + 10
    ctx_hi = jnp.asarray(rul["test_hi_context"])                 # (n_test, ctx, 1)
    covc = jnp.broadcast_to(jnp.asarray(cov)[:args.ctx][None, :, None], ctx_hi.shape)
    samples = himodel.sample(jr.PRNGKey(2), ctx_hi, covc, nfe=hicfg.flow_nfe,
                             n_steps_pred=horizon, n_samples=hicfg.flow_n_samples)  # (S,n,H,1)
    hi_fore = np.asarray(samples[..., 0]).mean(axis=0)            # (n_test, H) ensemble mean
    rul_pred = np.array([derive_rul(hi_fore[i], rul["threshold"], cap=args.cap)
                         for i in range(hi_fore.shape[0])], dtype=float)
    rul_true = np.minimum(rul["rul_true"], args.cap)
    rmse = float(np.sqrt(np.mean((rul_pred - rul_true) ** 2)))
    print(f"[cmapss {args.subset} RUL cap={args.cap}] RMSE={rmse:.2f} cycles  "
          f"Score={nasa_score(rul_pred, rul_true):.1f}")

if __name__ == "__main__":
    main()
