"""Verify the RNG handling of the AFM/FM samplers (reviewer rebuttal).

This script answers, empirically and on CPU, the reviewer's concern that the
autoregressive sampler may not give each (sample, batch, step) combination an
independent RNG stream.

It does three things:

  1. NOISE-STREAM AUDIT (model-free, deterministic):
     Reconstructs the *exact* key bookkeeping of the old sampler (per-sample key
     broadcast over the batch via ``in_axes=None``) and the new/fixed sampler
     (an independent key per (sample, batch) pair, split per step), then checks
     independence/sharing along each of the three axes.

  2. SAMPLER CROSS-CHECK:
     Confirms the in-repo ``model.sample`` (the fixed code) reproduces the
     standalone "new" sampler bit-for-bit for a fixed key.

  3. METRIC INVARIANCE:
     Trains a small AFM on a dynamical system, then computes CRPS and NRMSE with
     the old vs new sampler across many top-level seeds, and reports whether the
     two differ by more than Monte-Carlo error. Because both samplers draw from
     the same per-series marginal predictive law (the old one only adds spurious
     *cross-series* correlation), and CRPS/NRMSE are marginal functionals
     averaged over series/time/channel, the reported numbers must coincide in
     expectation.

Run:  PYTHONPATH=src python verify_rng_streams.py [n_seeds]
"""

import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import flax.nnx as nnx
import optax
import diffrax as dfx

jax.config.update("jax_platform_name", "cpu")

from afm.models.afm import AutoregressiveFlowMatching
from afm.systems.systems import create_system
from afm.evaluation.metrics import compute_crps, compute_nrmse


# --------------------------------------------------------------------------- #
# Problem size (modest, CPU-friendly; the invariance does not depend on size)
# --------------------------------------------------------------------------- #
DIM_X = 2
DIM_C = 1
CONTEXT_LEN = 25
N_STEPS_PRED = 12
NFE = 15
N_SAMPLES = 30
BATCH = 8
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
HIDDEN = 64


# --------------------------------------------------------------------------- #
# 1. NOISE-STREAM AUDIT  (replicates the two key-handling schemes exactly)
# --------------------------------------------------------------------------- #
def collect_base_noise(scheme, key, n_samples, batch, n_steps, dim):
    """Return the base-noise draws x0 of shape (S, B, steps, dim).

    Mirrors *exactly* how each sampler derives its keys, so the result reflects
    what the real sampler feeds into the ODE at each step.
    """
    out = np.zeros((n_samples, batch, n_steps, dim))

    if scheme == "old":
        # per-sample key, broadcast across the batch (in_axes=None on the key)
        key_samples = jrandom.split(key, n_samples)
        for s in range(n_samples):
            for b in range(batch):
                k = key_samples[s]                       # SAME k for every b
                for step in range(n_steps):
                    out[s, b, step] = np.asarray(jrandom.normal(k, (dim,)))
                    k, _ = jrandom.split(k)              # per-step split
    elif scheme == "new":
        # independent key per (sample, batch) pair
        grid = jrandom.split(key, n_samples * batch)
        grid = grid.reshape(n_samples, batch, *grid.shape[1:])
        for s in range(n_samples):
            for b in range(batch):
                k = grid[s, b]
                for step in range(n_steps):
                    out[s, b, step] = np.asarray(jrandom.normal(k, (dim,)))
                    k, _ = jrandom.split(k)
    else:
        raise ValueError(scheme)
    return out


def audit_axis_independence():
    key = jrandom.PRNGKey(0)
    print("=" * 72)
    print("1. NOISE-STREAM AUDIT")
    print("=" * 72)

    for scheme in ("old", "new"):
        x0 = collect_base_noise(scheme, key, N_SAMPLES, BATCH, N_STEPS_PRED, DIM_X)

        # batch independence: do batch elements within a sample share noise?
        batch_shared = bool(np.allclose(x0, x0[:, :1, :, :]))  # all b == b0 ?

        # sample independence: are different sample indices distinct?
        samples_distinct = not np.allclose(x0[0], x0[1])

        # step independence: are consecutive steps distinct?
        steps_distinct = not np.allclose(x0[:, :, 0, :], x0[:, :, 1, :])

        print(f"\n  scheme = {scheme!r}")
        print(f"    base noise identical across BATCH within a sample : {batch_shared}")
        print(f"    base noise distinct across SAMPLES                : {samples_distinct}")
        print(f"    base noise distinct across STEPS                  : {steps_distinct}")

    # Assertions encoding the reviewer's requirement.
    x_old = collect_base_noise("old", key, N_SAMPLES, BATCH, N_STEPS_PRED, DIM_X)
    x_new = collect_base_noise("new", key, N_SAMPLES, BATCH, N_STEPS_PRED, DIM_X)
    assert np.allclose(x_old, x_old[:, :1]), "expected OLD to share noise across batch"
    assert not np.allclose(x_new, x_new[:, :1]), "expected NEW to differ across batch"
    # all three axes independent in the new scheme:
    flat = x_new.reshape(-1, DIM_X)
    uniq = np.unique(np.round(flat, 8), axis=0)
    assert uniq.shape[0] == flat.shape[0], "NEW base-noise draws are not all distinct"
    print("\n  -> OLD shares base noise across the batch; NEW gives every")
    print("     (sample, batch, step) an independent draw.  [verdict confirmed]")


# --------------------------------------------------------------------------- #
# Standalone samplers (call the real model; reproduce each key scheme exactly)
# --------------------------------------------------------------------------- #
def make_predict_single(model):
    def predict_single(key, history, covariates, nfe, n_steps_pred):
        def vf(t, x, args):
            h, c = args
            return model(x, h, c, jnp.array([t]))

        term = dfx.ODETerm(vf)
        solver = dfx.Euler()
        dt = 1.0 / nfe

        predictions = []
        current_history = history
        context_length = history.shape[0]
        for _ in range(n_steps_pred):
            context = current_history[-context_length:]
            x0 = jrandom.normal(key, shape=context.shape[-1:])
            x_next = dfx.diffeqsolve(
                term, solver, 0.0, 1.0, dt, x0,
                args=(context, covariates),
                saveat=dfx.SaveAt(t1=True),
            ).ys
            x_next = jnp.squeeze(x_next, axis=-2)
            predictions.append(x_next)
            current_history = jnp.concatenate([current_history, x_next[None]], axis=0)
            key, _ = jrandom.split(key)
        return jnp.vstack(predictions)

    return predict_single


def sample_old(model, key, history, covariates, nfe, n_steps_pred, n_samples):
    """Original sampler: per-sample key broadcast across the batch."""
    ps = make_predict_single(model)
    key_samples = jrandom.split(key, n_samples)
    f = jax.vmap(ps, in_axes=(None, 0, 0, None, None))   # batch (key broadcast)
    f = jax.vmap(f, in_axes=(0, None, None, None, None))  # samples
    return f(key_samples, history, covariates, nfe, n_steps_pred)


def sample_new(model, key, history, covariates, nfe, n_steps_pred, n_samples):
    """Fixed sampler: independent key per (sample, batch) pair."""
    ps = make_predict_single(model)
    batch = history.shape[0]
    grid = jrandom.split(key, n_samples * batch)
    grid = grid.reshape(n_samples, batch, *grid.shape[1:])
    f = jax.vmap(ps, in_axes=(0, 0, 0, None, None))       # batch (key mapped)
    f = jax.vmap(f, in_axes=(0, None, None, None, None))  # samples
    return f(grid, history, covariates, nfe, n_steps_pred)


# --------------------------------------------------------------------------- #
# Data + a small AFM, briefly trained so the numbers are meaningful
# --------------------------------------------------------------------------- #
def make_trajectories(key, n_traj, n_t):
    system = create_system("lotka_volterra")
    ts = jnp.linspace(0.0, 20.0, n_t)

    def solve(x0):
        return dfx.diffeqsolve(
            system.vector_field, system.solver, ts[0], ts[-1], 0.01, x0,
            args=system.params, saveat=dfx.SaveAt(ts=ts),
        ).ys

    x0 = system.initial_state(key, n_traj)
    return jax.vmap(solve)(x0), ts


def afm_loss(model, rng, history, covariates, x_next):
    bs = history.shape[0]
    rng_t, rng_x0 = jrandom.split(rng)
    t = jrandom.uniform(rng_t, (bs,), minval=0.0, maxval=1.0)
    x0 = jrandom.normal(rng_x0, shape=x_next.shape)
    z = (1 - t[:, None]) * x0 + t[:, None] * x_next
    pred_flow = jax.vmap(model)(z, history, covariates, t)
    return jnp.mean(jnp.square(pred_flow - (x_next - x0)))


def train_model(model, traj, cov, n_t, steps=400, lr=1e-3):
    opt = nnx.Optimizer(model, optax.adam(lr))
    grad_fn = nnx.value_and_grad(afm_loss)
    rng = jrandom.PRNGKey(123)
    n_traj = traj.shape[0]
    last = None
    for i in range(steps):
        rng, kb, ki, kt = jrandom.split(rng, 4)
        # sample random single-step (context -> next) windows
        traj_idx = jrandom.randint(kb, (BATCH,), 0, n_traj)
        start = jrandom.randint(ki, (BATCH,), 0, n_t - CONTEXT_LEN - 1)
        hist = jnp.stack([traj[t, s:s + CONTEXT_LEN] for t, s in zip(traj_idx.tolist(), start.tolist())])
        nxt = jnp.stack([traj[t, s + CONTEXT_LEN] for t, s in zip(traj_idx.tolist(), start.tolist())])
        cov_b = jnp.stack([cov[s:s + CONTEXT_LEN] for s in start.tolist()])
        loss, grads = grad_fn(model, kt, hist, cov_b, nxt)
        opt.update(grads)
        last = float(loss)
    return last


# --------------------------------------------------------------------------- #
# 3. METRIC INVARIANCE
# --------------------------------------------------------------------------- #
def metric_invariance(model, history, covariates, ground_truth):
    print("\n" + "=" * 72)
    print("3. METRIC INVARIANCE  (old vs new sampler, across seeds)")
    print("=" * 72)

    def eval_once(sampler, seed):
        preds = sampler(model, jrandom.PRNGKey(seed), history, covariates,
                        NFE, N_STEPS_PRED, N_SAMPLES)              # (S,B,T,C)
        crps = float(compute_crps(preds, ground_truth))
        nrmse = float(compute_nrmse(preds, ground_truth))
        return crps, nrmse

    # warm-up / compile + timing probe
    t0 = time.time()
    eval_once(sample_old, 0)
    eval_once(sample_new, 0)
    print(f"  (compile + 1 eval each: {time.time() - t0:.1f}s; running {N_SEEDS} seeds)")

    old = np.array([eval_once(sample_old, 1000 + s) for s in range(N_SEEDS)])
    new = np.array([eval_once(sample_new, 1000 + s) for s in range(N_SEEDS)])

    def report(name, col):
        o, n = old[:, col], new[:, col]
        mo, so = o.mean(), o.std(ddof=1)
        mn, sn = n.mean(), n.std(ddof=1)
        se = float(np.sqrt(so**2 / N_SEEDS + sn**2 / N_SEEDS))   # SE of the diff
        diff = abs(mo - mn)
        verdict = "within MC error" if diff <= 2 * se else "DIFFERENT (>2 SE)"
        print(f"\n  {name}")
        print(f"    old : {mo:.6f} +/- {so:.6f}   (mean +/- sd over {N_SEEDS} seeds)")
        print(f"    new : {mn:.6f} +/- {sn:.6f}")
        print(f"    |old-new| = {diff:.6f}   SE(diff) = {se:.6f}   -> {verdict}")
        return diff <= 2 * se

    ok_crps = report("CRPS", 0)
    ok_nrmse = report("NRMSE", 1)
    return ok_crps and ok_nrmse


def main():
    print(f"config: dim={DIM_X} ctx={CONTEXT_LEN} steps={N_STEPS_PRED} nfe={NFE} "
          f"samples={N_SAMPLES} batch={BATCH} seeds={N_SEEDS}\n")

    audit_axis_independence()

    # --- build data + model ---
    n_t = CONTEXT_LEN + N_STEPS_PRED + 5
    traj, ts = make_trajectories(jrandom.PRNGKey(7), n_traj=64, n_t=n_t)
    cov = ts[:, None]  # (n_t, 1) time-stamp covariate

    model = AutoregressiveFlowMatching(
        dim_x=DIM_X, dim_c=DIM_C, dim_h=HIDDEN, dim_t=16, dim_hidden=HIDDEN,
        hidden_blocks=3, n_rnn_layers=3, rnn_hidden=HIDDEN, rngs=nnx.Rngs(0),
    )
    print("\n" + "=" * 72)
    print("2. TRAIN small AFM + SAMPLER CROSS-CHECK")
    print("=" * 72)
    final_loss = train_model(model, traj, cov, n_t)
    print(f"  trained AFM (final single-step FM loss = {final_loss:.4f})")

    # build a test batch: history window + true future
    starts = list(range(0, BATCH))
    history = jnp.stack([traj[i, 0:CONTEXT_LEN] for i in starts])         # (B,ctx,dim)
    ground_truth = jnp.stack(
        [traj[i, CONTEXT_LEN:CONTEXT_LEN + N_STEPS_PRED] for i in starts]  # (B,T,dim)
    )
    # covariates are mapped over the batch axis inside sample(), so batch them
    covariates = jnp.repeat(cov[:CONTEXT_LEN][None], BATCH, axis=0)  # (B,ctx,1)

    # cross-check: in-repo model.sample (fixed) == standalone sample_new
    k = jrandom.PRNGKey(42)
    repo = model.sample(k, history, covariates, nfe=NFE,
                        n_steps_pred=N_STEPS_PRED, n_samples=N_SAMPLES)
    mine = sample_new(model, k, history, covariates, NFE, N_STEPS_PRED, N_SAMPLES)
    match = bool(jnp.allclose(repo, mine))
    print(f"  in-repo model.sample == standalone NEW sampler : {match}")
    assert match, "in-repo fixed sampler does not match the standalone NEW sampler"

    ok = metric_invariance(model, history, covariates, ground_truth)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  - OLD sampler shared base noise across the batch within a sample index;")
    print("    NEW sampler gives every (sample, batch, step) an independent stream.")
    print("  - In-repo model.sample now matches the NEW sampler.")
    print(f"  - CRPS & NRMSE unchanged within Monte-Carlo error: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
