"""Candidate I -- Phase 3 error analysis of Candidate G (where does G lose?).

For each slice we report WAPE of F and G, G's share of total absolute error, and the
"correction gap": mean of log(actual/F) - log(g_factor), i.e. how far G's multiplicative
factor was from the factor that would have been exactly right for that target
(positive = G under-corrected upward / over-corrected downward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOGOTA_TZ = "America/Bogota"


def enrich_for_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    local = out["target_at"].dt.tz_convert(BOGOTA_TZ)
    out["target_hour_local"] = local.dt.hour
    out["is_weekend"] = local.dt.dayofweek.isin([5, 6])
    oracle = np.log(np.clip(out["actual"], 1e-3, None) / np.clip(out["pred_F"], 1e-3, None))
    out["oracle_log_factor"] = oracle
    out["g_log_factor"] = np.log(out["g_factor"].clip(lower=1e-3))
    out["correction_gap"] = out["oracle_log_factor"] - out["g_log_factor"]
    out["drift_direction"] = pd.cut(out["oracle_log_factor"], [-np.inf, -0.15, 0.15, np.inf], labels=["down(<0.86x)", "flat", "up(>1.16x)"])
    rel = out["pred_F"] / out.groupby("station_id")["pred_F"].transform("mean")
    out["demand_level"] = pd.qcut(rel, 4, labels=["Q1_low", "Q2", "Q3", "Q4_peak"])
    if "day" in out and "ssday1" in out:
        speed = np.abs(np.log(np.clip(out["day"], 1e-3, None)) - np.log(np.clip(out["ssday1"], 1e-3, None)))
        out["drift_speed"] = pd.cut(speed, [-np.inf, 0.05, 0.15, np.inf], labels=["slow(<5%/d)", "medium", "fast(>16%/d)"])
    return out


def slice_table(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    f = frame.assign(abs_F=(frame["pred_F"] - frame["actual"]).abs(), abs_G=(frame["pred_G"] - frame["actual"]).abs())
    total_g = f["abs_G"].sum()
    g = f.groupby(by, observed=True).agg(
        rows=("actual", "size"),
        actual_sum=("actual", "sum"),
        abs_F=("abs_F", "sum"),
        abs_G=("abs_G", "sum"),
        correction_gap=("correction_gap", "mean"),
        abs_correction_gap=("correction_gap", lambda v: float(np.mean(np.abs(v)))),
    )
    g["wape_F"] = g["abs_F"] / g["actual_sum"]
    g["wape_G"] = g["abs_G"] / g["actual_sum"]
    g["G_gain_vs_F_pp"] = 100 * (g["wape_F"] - g["wape_G"])
    g["share_of_G_abs_error"] = g["abs_G"] / total_g
    return g[["rows", "wape_F", "wape_G", "G_gain_vs_F_pp", "share_of_G_abs_error", "correction_gap", "abs_correction_gap"]]
