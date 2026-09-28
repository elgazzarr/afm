"""Statistical baselines for Table 4 (IJF revision R2).

Rebuilds the AFM GluonTS evaluation windows and evaluator (recovered from git
``f4131fe^:data/gluonts_loaders.py`` and ``f4131fe^:trainers/eval_gluonts.py``)
and scores ``statsforecast`` baselines on exactly those windows.

Evaluation modes
----------------
afm     The AFM harness: ``MultivariateGrouper(num_test_dates=W)`` test windows,
        ``MultivariateEvaluator`` with quantiles 0.05, 0.10, ..., 0.95; the Table 4
        metric is ``mean_wQuantileLoss`` (per-dimension wQL averaged over dims).
tsflow  The protocol of the quoted rows (Kollovieh et al., TSDiff/TSFlow): each
        series-window scored as a univariate forecast with the GluonTS
        ``Evaluator`` at quantiles 0.1, ..., 0.9. Used only to validate the harness
        against the quoted numbers.

Forecasts are Gaussian prediction intervals at levels 10, 20, ..., 90 mapped to
quantiles (1 -/+ L/100)/2, with the point forecast as the median, wrapped as
GluonTS ``QuantileForecast`` objects.

Usage (from the repo root, with the baseline env):
    baselines/.venv/bin/python baselines/r2_statistical_baselines.py validate
    baselines/.venv/bin/python baselines/r2_statistical_baselines.py naive
    baselines/.venv/bin/python baselines/r2_statistical_baselines.py select-k --dataset electricity_nips
    baselines/.venv/bin/python baselines/r2_statistical_baselines.py run --model mstl --dataset solar_nips
    baselines/.venv/bin/python baselines/r2_statistical_baselines.py run --model dhr --dataset solar_nips --k1 4 --k2 4
Add ``--n-series 10 --n-windows 1`` to ``run`` for a timing subset.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gluonts  # noqa: E402
import statsforecast  # noqa: E402
from gluonts.dataset.field_names import FieldName  # noqa: E402
from gluonts.dataset.multivariate_grouper import MultivariateGrouper  # noqa: E402
from gluonts.dataset.repository import get_dataset  # noqa: E402
from gluonts.dataset.split import split  # noqa: E402
from gluonts.dataset.util import period_index  # noqa: E402
from gluonts.evaluation import Evaluator, MultivariateEvaluator  # noqa: E402
from gluonts.model.forecast import QuantileForecast  # noqa: E402
from statsforecast import StatsForecast  # noqa: E402
from statsforecast.models import MSTL, AutoARIMA, AutoETS, Naive, SeasonalNaive  # noqa: E402

LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
AFM_QUANTILES = (np.arange(20) / 20.0)[1:]  # 0.05, ..., 0.95 (P:trainers/eval_gluonts.py:93)
TSFLOW_QUANTILES = np.linspace(0.1, 0.9, 9)  # GluonTS Evaluator default, as in TSDiff/TSFlow
# Forecast domain Omega per dataset (paper Table 3): clip quantiles into it.
DOMAIN = {
    "electricity_nips": (0.0, None),
    "solar_nips": (0.0, None),
    "traffic_nips": (0.0, 1.0),
    "exchange_rate_nips": (0.0, None),
}
OUT_DIR = Path("results/r2")


# ----------------------------------------------------------------------------
# Data and windows
# ----------------------------------------------------------------------------
def _quantile_keys() -> List[str]:
    # Keys must match gluonts' Quantile names ("0.1", not "0.10"), or lookups return NaN.
    lo = [f"{round((1 - L / 100) / 2, 2):g}" for L in reversed(LEVELS)]  # 0.05 ... 0.45
    hi = [f"{round((1 + L / 100) / 2, 2):g}" for L in LEVELS]  # 0.55 ... 0.95
    return lo + ["0.5"] + hi


QKEYS = _quantile_keys()


def _to_dataframe(input_label) -> pd.DataFrame:
    """Copied from P:data/gluonts_loaders.py (AFM target construction)."""
    start = input_label[0][FieldName.START]
    full_target = np.concatenate([e[FieldName.TARGET] for e in input_label], axis=-1)
    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    return pd.DataFrame(full_target.transpose(), index=index)


class Windows:
    """Test windows of a GluonTS *_nips dataset, per series and grouped as in AFM."""

    def __init__(self, name: str):
        self.name = name
        ds = get_dataset(name, regenerate=False)
        self.freq = ds.metadata.freq
        self.H = int(ds.metadata.prediction_length)
        self.train = list(ds.train)
        self.test = list(ds.test)
        self.N = len(self.train)
        self.W = len(self.test) // self.N
        assert self.W * self.N == len(self.test)
        self.card = int(ds.metadata.feat_static_cat[0].cardinality)
        self._ds = ds

    def entry(self, w: int, i: int) -> dict:
        return self.test[w * self.N + i]

    def history(self, w: int, i: int) -> np.ndarray:
        y = np.asarray(self.entry(w, i)[FieldName.TARGET], dtype=float)[: -self.H]
        assert np.isfinite(y).all(), f"non-finite history in {self.name} w={w} i={i}"
        return y

    def future(self, w: int, i: int) -> np.ndarray:
        return np.asarray(self.entry(w, i)[FieldName.TARGET], dtype=float)[-self.H:]

    def origin(self, w: int, i: int) -> pd.Period:
        e = self.entry(w, i)
        return e[FieldName.START] + (len(e[FieldName.TARGET]) - self.H)

    def afm_targets(self) -> Tuple[List[pd.DataFrame], List[pd.Period]]:
        """Grouped test windows exactly as P:data/gluonts_loaders.py:131-200."""
        grouper = MultivariateGrouper(
            num_test_dates=self.W, max_target_dim=min(2000, self.card)
        )
        grouped = grouper(self._ds.test)
        _, template = split(grouped, offset=-self.H)
        instances = list(template.generate_instances(self.H))
        assert len(instances) == self.W
        targets = [_to_dataframe(x) for x in instances]
        starts = [label[FieldName.START] for _, label in instances]
        # The grouped labels must equal the per-series futures, in dataset order.
        for w, (_, label) in enumerate(instances):
            per_series = np.stack([self.future(w, i) for i in range(self.N)])
            np.testing.assert_allclose(label[FieldName.TARGET], per_series, rtol=0, atol=1e-5)
            assert all(self.origin(w, i) == starts[w] for i in range(self.N))
        return targets, starts


# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------
def fourier_terms(t: np.ndarray, period: int, K: int) -> np.ndarray:
    """sin/cos pairs k=1..K of absolute time index t (hours since the epoch)."""
    k = np.arange(1, K + 1)
    ang = 2 * np.pi * np.outer(t, k) / period
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def dhr_exog(t: np.ndarray, k1: int, k2: int) -> np.ndarray:
    return np.concatenate([fourier_terms(t, 24, k1), fourier_terms(t, 168, k2)], axis=1)


def make_model(spec: dict):
    kind = spec["model"]
    if kind == "naive":
        return Naive()
    if kind == "sn":
        return SeasonalNaive(season_length=spec["m"])
    if kind == "ets":
        return AutoETS(season_length=spec["m"])
    if kind == "arima":
        return AutoARIMA(season_length=spec["m"])
    if kind == "mstl":
        return MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))
    if kind == "dhr":
        return AutoARIMA(season_length=1)
    raise ValueError(kind)


def forecast_quantiles(
    wins: Windows,
    spec: dict,
    cells: Sequence[Tuple[int, int]],
    history: Optional[int],
    point_only: bool = False,
    n_jobs: int = -1,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Return {(w, i): array (len(QKEYS)+1, H)} with rows QKEYS then 'mean'."""
    H = wins.H
    rows, xrows = [], []
    for w, i in cells:
        y = wins.history(w, i)
        if history is not None:
            y = y[-history:]
        uid = f"{w}_{i}"
        df = pd.DataFrame({"unique_id": uid, "ds": np.arange(len(y)), "y": y})
        if spec["model"] == "dhr":
            t0 = wins.origin(w, i).ordinal - len(y)  # hours since epoch of first point
            t = t0 + np.arange(len(y) + H)
            X = dhr_exog(t, spec["k1"], spec["k2"])
            cols = [f"x{j}" for j in range(X.shape[1])]
            df[cols] = X[: len(y)]
            xf = pd.DataFrame(X[len(y):], columns=cols)
            xf.insert(0, "ds", np.arange(len(y), len(y) + H))
            xf.insert(0, "unique_id", uid)
            xrows.append(xf)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    X_df = pd.concat(xrows, ignore_index=True) if xrows else None
    model = make_model(spec)
    sf = StatsForecast(models=[model], freq=1, n_jobs=n_jobs)
    fc = sf.forecast(df=df, h=H, X_df=X_df, level=None if point_only else LEVELS)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    alias = repr(model)
    out = {}
    lo, hi = DOMAIN.get(wins.name, (None, None))
    for uid, g in fc.groupby("unique_id", sort=False):
        g = g.sort_values("ds")
        point = g[alias].to_numpy()
        if point_only:
            q = np.tile(point, (len(QKEYS), 1))
        else:
            lows = [g[f"{alias}-lo-{L}"].to_numpy() for L in reversed(LEVELS)]
            highs = [g[f"{alias}-hi-{L}"].to_numpy() for L in LEVELS]
            q = np.stack(lows + [point] + highs)
        if spec.get("clip", True):
            q = np.clip(q, lo, hi)
            point = np.clip(point, lo, hi)
        w, i = map(int, uid.split("_"))
        out[(w, i)] = np.vstack([q, point[None]])
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_afm(wins: Windows, fc: Dict[Tuple[int, int], np.ndarray]) -> dict:
    targets, starts = wins.afm_targets()
    forecasts = []
    for w in range(wins.W):
        arr = np.stack([fc[(w, i)] for i in range(wins.N)], axis=-1)  # (Q+1, H, D)
        forecasts.append(QuantileForecast(arr, starts[w], QKEYS + ["mean"]))
    ev = MultivariateEvaluator(quantiles=AFM_QUANTILES, num_workers=None)
    agg, _ = ev(iter(targets), iter(forecasts), num_series=len(targets))
    return {"CRPS": float(agg["mean_wQuantileLoss"]), "ND": float(agg["ND"])}


def evaluate_tsflow(wins: Windows, fc: Dict[Tuple[int, int], np.ndarray]) -> dict:
    targets, forecasts = [], []
    for w in range(wins.W):
        for i in range(wins.N):
            e = wins.entry(w, i)
            y = np.asarray(e[FieldName.TARGET], dtype=float)
            idx = period_index({FieldName.START: e[FieldName.START], FieldName.TARGET: y})
            targets.append(pd.DataFrame(y, index=idx))
            forecasts.append(QuantileForecast(fc[(w, i)], wins.origin(w, i), QKEYS + ["mean"]))
    ev = Evaluator(quantiles=TSFLOW_QUANTILES, num_workers=None)
    agg, _ = ev(iter(targets), iter(forecasts), num_series=len(targets))
    return {"CRPS": float(agg["mean_wQuantileLoss"]), "ND": float(agg["ND"])}


def versions() -> dict:
    import utilsforecast

    return {
        "python": platform.python_version(),
        "gluonts": gluonts.__version__,
        "statsforecast": statsforecast.__version__,
        "utilsforecast": utilsforecast.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "machine": f"{platform.machine()} {os.cpu_count()} cpus",
    }


def save(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    payload = {**payload, "versions": versions(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(payload, indent=2))
    return path


def all_cells(wins: Windows, n_series: Optional[int] = None, n_windows: Optional[int] = None):
    ws = range(wins.W if n_windows is None else n_windows)
    ss = range(wins.N if n_series is None else n_series)
    return [(w, i) for w in ws for i in ss]


def run_one(wins: Windows, spec: dict, history: Optional[int], point_only=False, modes=("afm", "tsflow")):
    t = time.time()
    fc = forecast_quantiles(wins, spec, all_cells(wins), history, point_only=point_only)
    fit_s = time.time() - t
    res = {"spec": spec, "history": history, "fit_seconds": round(fit_s, 1)}
    if "afm" in modes:
        res["afm"] = evaluate_afm(wins, fc)
    if "tsflow" in modes:
        res["tsflow"] = evaluate_tsflow(wins, fc)
    return res


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_validate(args):
    """Reproduce quoted TSFlow rows: SN (deterministic) and ETS/ARIMA on Exchange."""
    out = {}
    # TSFlow's season lengths (tsflow/utils/variables.py: {"H": 24, "D": 30, "B": 30});
    # m=30 reproduces the quoted SN-Exchange 0.013 exactly (m=5 gives 0.0107).
    sn_m = {"exchange_rate_nips": 30, "electricity_nips": 24, "solar_nips": 24, "traffic_nips": 24}
    for name in args.datasets:
        wins = Windows(name)
        spec = {"model": "sn", "m": sn_m[name], "clip": False}
        out[f"SN/{name}"] = run_one(wins, spec, history=None, point_only=True)
        print(name, "SN", out[f"SN/{name}"], flush=True)
        if name == "exchange_rate_nips":
            for model in ("ets", "arima"):
                for hist in (2500, None):
                    spec = {"model": model, "m": 5, "clip": False}
                    key = f"{model.upper()}/{name}/hist={hist}"
                    out[key] = run_one(wins, spec, history=hist)
                    print(key, out[key], flush=True)
    print("saved", save("validate", {"results": out}))


def cmd_naive(args):
    wins = Windows("exchange_rate_nips")
    res = {}
    for hist in (None, 2500):
        res[f"hist={hist}"] = run_one(wins, {"model": "naive"}, history=hist)
        print(hist, res[f"hist={hist}"], flush=True)
    print("saved", save("naive_exchange", {"results": res}))


def _dhr_aicc(i: int, y: np.ndarray, t0: int, k1: int, k2: int):
    X = dhr_exog(t0 + np.arange(len(y)), k1, k2)
    mod = AutoARIMA(season_length=1).fit(y, X).model_
    # arma = (p, q, P, Q, m, d, D)
    return i, k1, k2, float(mod["aicc"]), (int(mod["arma"][0]), int(mod["arma"][1]), int(mod["arma"][5]))


def cmd_select_k(args):
    """Choose (K1, K2) for DHR by AICc summed over a random subset of series."""
    from concurrent.futures import ProcessPoolExecutor

    wins = Windows(args.dataset)
    rng = np.random.default_rng(args.seed)
    series = sorted(rng.choice(wins.N, size=min(args.n_series, wins.N), replace=False).tolist())
    w = 0
    grid = list(itertools.product(args.k_grid, args.k_grid))
    jobs = []
    for i in series:
        y = wins.history(w, i)[-args.history:]
        t0 = wins.origin(w, i).ordinal - len(y)
        jobs += [(i, y, t0, k1, k2) for (k1, k2) in grid]

    t = time.time()
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_dhr_aicc, *zip(*jobs)))
    df = pd.DataFrame(rows, columns=["series", "k1", "k2", "aicc", "pqd"])
    table = df.groupby(["k1", "k2"])["aicc"].sum().sort_values()
    best = table.index[0]
    res = {
        "dataset": args.dataset, "series": series, "window": w, "history": args.history,
        "k_grid": args.k_grid, "aicc_sum": {f"{k1},{k2}": v for (k1, k2), v in table.items()},
        "best": {"k1": int(best[0]), "k2": int(best[1])},
        "d_counts": df["pqd"].map(lambda x: x[2]).value_counts().to_dict(),
        "seconds": round(time.time() - t, 1),
    }
    print(table.head(10))
    print("best", res["best"], "seconds", res["seconds"])
    print("saved", save(f"selectk_{args.dataset}", res))


def cmd_run(args):
    wins = Windows(args.dataset)
    spec = {"model": args.model}
    if args.model == "dhr":
        spec.update(k1=args.k1, k2=args.k2)
    cells = all_cells(wins, args.n_series, args.n_windows)
    t = time.time()
    fc = forecast_quantiles(wins, spec, cells, args.history)
    secs = time.time() - t
    res = {"dataset": args.dataset, "spec": spec, "history": args.history,
           "n_cells": len(cells), "seconds": round(secs, 1),
           "seconds_per_cell": round(secs / len(cells), 3),
           "est_full_seconds": round(secs / len(cells) * wins.W * wins.N, 0)}
    if args.n_series is None and args.n_windows is None:
        res["afm"] = evaluate_afm(wins, fc)
        res["tsflow"] = evaluate_tsflow(wins, fc)
        tag = f"{args.model}_{args.dataset}"
    else:
        tag = f"timing_{args.model}_{args.dataset}"
    print(res, flush=True)
    print("saved", save(tag, res))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--datasets", nargs="+",
                   default=["exchange_rate_nips", "electricity_nips", "solar_nips", "traffic_nips"])
    sub.add_parser("naive")
    k = sub.add_parser("select-k")
    k.add_argument("--dataset", required=True)
    k.add_argument("--n-series", type=int, default=20)
    k.add_argument("--k-grid", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    k.add_argument("--history", type=int, default=1344)
    k.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=["mstl", "dhr"], required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--history", type=int, default=1344)
    r.add_argument("--k1", type=int)
    r.add_argument("--k2", type=int)
    r.add_argument("--n-series", type=int)
    r.add_argument("--n-windows", type=int)
    args = p.parse_args()
    {"validate": cmd_validate, "naive": cmd_naive, "select-k": cmd_select_k, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
