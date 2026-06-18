from __future__ import annotations
import os

BATTERY_URL = "https://ti.arc.nasa.gov/c/5/"  # NASA PCoE "Battery Data Set" (.mat zip); mirror: kaggle.com/datasets/patrickfleith/nasa-battery-dataset
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"   # NASA PCoE "Turbofan Engine Degradation Simulation" (CMAPSSData.zip); mirror: kaggle.com/datasets/behrad3d/nasa-cmaps

def _need(path: str, url: str, what: str) -> None:
    raise FileNotFoundError(
        f"Required file not found: {path}\n"
        f"Download '{what}' from: {url}\n"
        f"(NASA PCoE may require manual download; Kaggle mirrors exist — see comments in download.py.)\n"
        f"Place the extracted file at the path above, then re-run."
    )

def ensure_cmapss_files(subset: str, data_dir: str = "data/cmapss", allow_download: bool = True) -> dict[str, str]:
    subset = subset.upper()
    paths = {
        "train": os.path.join(data_dir, f"train_{subset}.txt"),
        "test": os.path.join(data_dir, f"test_{subset}.txt"),
        "rul": os.path.join(data_dir, f"RUL_{subset}.txt"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        # Auto-download is intentionally not implemented for NASA's interstitial-gated host;
        # rely on the manual/mirror instructions. allow_download kept for future mirror support.
        _need(missing[0], CMAPSS_URL, f"C-MAPSS {subset} (train/test/RUL .txt)")
    return paths

def ensure_battery_files(cells: list[str], data_dir: str = "data/battery", allow_download: bool = True) -> dict[str, str]:
    paths = {c: os.path.join(data_dir, f"{c}.mat") for c in cells}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        _need(missing[0], BATTERY_URL, "NASA Li-ion Battery Aging (B0005/6/7/18 .mat)")
    return paths
