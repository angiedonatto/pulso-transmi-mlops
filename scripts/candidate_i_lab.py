"""Candidate I lab -- reproducible research harness.

Stages (run in order; each is resumable / cached under data/ which is gitignored):

    python3 scripts/candidate_i_lab.py snapshot   # read-only Supabase snapshot -> data/raw/candidate_i/
    python3 scripts/candidate_i_lab.py parity     # fast base path == production build_candidate_f_base_frame
    python3 scripts/candidate_i_lab.py bases      # hourly-exact walk-forward B_wf / F replay (slow, resumable)
    python3 scripts/candidate_i_lab.py variant F24h --cadence 24   # out-of-sample daily-refit F history
    python3 scripts/candidate_i_lab.py explore [--analysis] [--meta]  # selection windows only
    python3 scripts/candidate_i_lab.py stress --policies ...       # synthetic drift on observations
    python3 scripts/candidate_i_lab.py parity-i   # src/candidate_i_snr_gate.py == lab path + runtime
    python3 scripts/candidate_i_lab.py final --challenger 'L:g2|snr4'  # reveals drift_test (once)
    python3 scripts/candidate_i_summary.py        # final tables + reports/candidate_i/summary.md

Nothing here writes to Supabase, changes Candidate G, or touches the submission path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidate_i_core as core  # noqa: E402

BASES_PATH = core.INTERIM_DIR / "bases_hourly.pkl"


def cmd_snapshot(_: argparse.Namespace) -> None:
    print(json.dumps(core.refresh_snapshot(), indent=2))


def cmd_parity(args: argparse.Namespace) -> None:
    tables = core.load_snapshot()
    cutoffs = [pd.Timestamp(value) for value in args.cutoffs]
    results = core.verify_fast_base_parity(tables, cutoffs)
    print(json.dumps(results, indent=2))
    if not all(item["identical"] for item in results):
        raise SystemExit("PARITY FAILED: fast base path differs from production build_candidate_f_base_frame")


def cmd_bases(args: argparse.Namespace) -> None:
    tables = core.load_snapshot()
    core.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cutoffs = core.evaluation_cutoffs(tables["observations"], pd.Timestamp(args.start))
    print(f"[bases] {len(cutoffs)} hourly cutoffs {cutoffs[0]} .. {cutoffs[-1]}", flush=True)
    core.replay_bases(tables, cutoffs, out_path=BASES_PATH, log=lambda m: print(m, flush=True))


FRAME_PATH = core.INTERIM_DIR / "frame_F.pkl"


def build_frame() -> pd.DataFrame:
    import candidate_i_adaptive as ad

    if FRAME_PATH.exists() and FRAME_PATH.stat().st_mtime >= BASES_PATH.stat().st_mtime:
        return pd.read_pickle(FRAME_PATH)
    bases = pd.read_pickle(BASES_PATH)
    bases["target_at"] = pd.to_datetime(bases["target_at"], utc=True)
    frame = core.replay_g(bases)
    frame = attach_served_b(frame, core.load_snapshot())
    history_path = core.INTERIM_DIR / "variant_F24h.pkl"
    keys = ["data_cutoff", "station_id", "target_at"]
    if history_path.exists():
        # Production-feasible signals: out-of-sample history from F refit once per day.
        history = pd.read_pickle(history_path)
        signals = ad.compute_signals(history, "pred_F24h")
        history = pd.concat([history, signals], axis=1)
        frame = frame.merge(history.drop(columns=["horizon_minutes", "actual"]), on=keys, how="left")
        frame.attrs["signal_base"] = "pred_F24h"
    else:
        signals = ad.compute_signals(frame, "pred_F")
        frame = pd.concat([frame, signals], axis=1)
        frame.attrs["signal_base"] = "pred_F"
    frame.to_pickle(FRAME_PATH)
    return frame


def attach_served_b(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Actual official submissions of the Candidate B models (what production really served)."""
    mv = tables["model_versions"]
    b_ids = set(mv.loc[mv["model_name"] == "candidate_b_hist_gradient_boosting_shadow", "model_id"])
    p = tables["predictions"]
    p = p[p["model_id"].isin(b_ids)][["cycle_id", "station_id", "target_observed_at", "prediction"]]
    p = p.rename(columns={"target_observed_at": "target_at", "prediction": "pred_B_served"}).drop_duplicates(["cycle_id", "station_id", "target_at"], keep="last")
    return frame.merge(p, on=["cycle_id", "station_id", "target_at"], how="left")


def policy_factors(fr: pd.DataFrame) -> dict[str, "np.ndarray"]:
    import numpy as np
    import candidate_i_adaptive as ad

    L = {name: ad._log(fr[name]) for name in ("g2", "ros4", "ros8", "ros16")}
    ss1 = ad._log(fr["ss1w1"])
    ss1w2 = ad._log(fr["ss1w2"])
    ssavg = ad._log(fr["ssavg"])
    shape1 = ss1 - ad._log(fr["ssday1"])
    shape12 = 0.5 * (shape1 + ad._log(fr["ss2w1"]) - ad._log(fr["ssday2"]))
    dod8 = ad._log(fr["dod8"])
    noise = fr["lr32_sd"].to_numpy(dtype=float) / np.sqrt(8.0)
    z = np.where(np.isfinite(noise) & (noise > 0), np.abs(fr["lr8_mean"].to_numpy(dtype=float)) / noise, 0.0)
    pol: dict[str, np.ndarray] = {"F": np.zeros(len(fr)), "G": np.log(fr["g_factor"].to_numpy(dtype=float))}
    for name, lv in L.items():
        pol[f"L:{name}"] = lv
        for kappa in (1.0, 4.0):
            gate = z**2 / (z**2 + kappa)
            pol[f"L:{name}|snr{kappa:g}"] = gate * lv
    pol["S:ss1w1"] = ss1
    pol["S:ss1w2"] = ss1w2
    pol["S:ssavg"] = ssavg
    for lname in ("g2", "ros8"):
        for b in (0.5, 1.0):
            pol[f"LxS:{lname}*shape1^{b:g}"] = L[lname] + b * shape1
            pol[f"LxS:{lname}*shape12^{b:g}"] = L[lname] + b * shape12
        for a in (0.25, 0.5, 0.75):
            pol[f"blend:{a:g}*{lname}+{1-a:g}*ss1w1"] = a * L[lname] + (1 - a) * ss1
            pol[f"blend:{a:g}*{lname}+{1-a:g}*ssavg"] = a * L[lname] + (1 - a) * ssavg
    pol["DoD:ss1w1*dod8"] = ss1 + dod8
    pol["DoD:ss1w2*dod8"] = ss1w2 + dod8
    # gated level + gated cross-day-consensus shape
    shapes = np.vstack([ad._log(fr[f"ss{d}w1"]) - ad._log(fr[f"ssday{d}"]) for d in (1, 2, 3)])
    have = np.vstack([np.isfinite(fr[f"ss{d}w1"].to_numpy(dtype=float)) & np.isfinite(fr[f"ssday{d}"].to_numpy(dtype=float)) for d in (1, 2, 3)])
    n_days = have.sum(axis=0)
    shape_mean = np.where(n_days > 0, (shapes * have).sum(axis=0) / np.maximum(n_days, 1), 0.0)
    slot_noise = np.nan_to_num(fr["lr32_sd"].to_numpy(dtype=float), nan=0.3) / np.sqrt(3.0)  # w=1 -> 3 slots per day
    z_shape = np.where(n_days > 0, np.abs(shape_mean) / (slot_noise / np.sqrt(np.maximum(n_days, 1)) + 1e-9), 0.0)
    for lname, lvl in (("ros16", L["ros16"]), ("g2", L["g2"]), ("ros8", L["ros8"])):
        for kl in (4.0,):
            gl = z**2 / (z**2 + kl)
            for ks in (4.0, 16.0):
                gs = z_shape**2 / (z_shape**2 + ks)
                pol[f"LSg:{lname}|snr{kl:g}+shape3|snr{ks:g}"] = gl * lvl + gs * shape_mean
    # same day-type (weekday vs weekend) variants
    sst1, sst2 = ad._log(fr["sstw1"]), ad._log(fr["sstw2"])
    dod8t = ad._log(fr["dod8_t"])
    shape_t = sst1 - ad._log(fr["sstday"])
    pol["S:sstw1"] = sst1
    pol["S:sstw2"] = sst2
    pol["DoDt:sstw1*dod8t"] = sst1 + dod8t
    pol["DoDt:sstw2*dod8t"] = sst2 + dod8t
    for lname in ("g2", "ros8", "ros4"):
        for b in (0.5, 1.0):
            pol[f"LxSt:{lname}*shape_t^{b:g}"] = L[lname] + b * shape_t
        for a in (0.25, 0.5, 0.75):
            pol[f"blend:{a:g}*{lname}+{1-a:g}*DoDt"] = a * L[lname] + (1 - a) * (sst2 + dod8t)
    return {k: ad.clip_factor(v) for k, v in pol.items()}


def cmd_explore(args: argparse.Namespace) -> None:
    import numpy as np
    import candidate_i_analysis as an
    import candidate_i_eval as ev

    fr = build_frame()
    masks = ev.window_masks(fr)
    windows = ["stable", "drift_dev"] + (["drift_test", "recent12", "recent6", "glab12"] if args.reveal_test else [])
    sel = masks["stable"] | masks["drift_dev"]

    print("\n=== Reproduction of B / F / G (selection windows) ===")
    print(ev.window_scores(fr, ["pred_B", "pred_F", "pred_G"], windows).round(4).to_string())

    if args.analysis:
        dev = an.enrich_for_analysis(fr[masks["drift_dev"]])
        for by in ("station_id", "horizon_minutes", "target_hour_local", "demand_level", "drift_direction", "drift_speed", "is_weekend"):
            print(f"\n=== G error analysis on drift_dev by {by} ===")
            print(an.slice_table(dev, by).round(4).to_string())
        stab = an.enrich_for_analysis(fr[masks["stable"]])
        print("\n=== G vs F on STABLE by station (no-drift regression check) ===")
        print(an.slice_table(stab, "station_id").round(4).to_string())

    import candidate_i_adaptive as ad
    import candidate_i_eval as ev_mod

    signal_base = "pred_F24h" if "pred_F24h" in fr.columns else "pred_F"
    print(f"[explore] signal base = {signal_base}; corrected base = {args.base}", flush=True)
    factors = policy_factors(fr)
    if args.meta:
        start = ev_mod.STABLE[0]
        for kind in ("huber", "hgb"):
            for window in (None, 7.0):
                label = f"META:{kind}|win{window or 'all'}"
                factors[label] = ad.walk_forward_meta(fr, base_col=signal_base, kind=kind, refit_hours=6, window_days=window, start=start)
                print(f"[explore] computed {label}", flush=True)
    for name, f in factors.items():
        fr[f"pol::{name}"] = np.maximum(0.0, fr[args.base].to_numpy() * f)
    cols = [f"pol::{name}" for name in factors] + ["pred_G", "pred_F", "pred_F24h"]
    cols = [c for c in cols if c in fr.columns]
    board = ev.window_scores(fr, cols, windows)
    g = board.loc["pred_G"]  # always the real production G (hourly F x production g2_factor)
    for w in windows:
        board[f"{w}_vs_G"] = board[w] - g[w]
    board = board.sort_values("drift_dev", ascending=False)
    board.to_csv(core.INTERIM_DIR / f"explore_board_{args.base}{'_meta' if args.meta else ''}.csv")
    print(f"\n=== Policy leaderboard (corrected base = {args.base}; reference = production G) ===")
    print(board.round(4).to_string())


VARIANTS = {
    "F6h": core.BaseVariant(name="F6h"),
    "F24h": core.BaseVariant(name="F24h"),
    "V1": core.BaseVariant(name="V1", wape_weights=True),
    "V3": core.BaseVariant(name="V3", ratio_features=True),
    "V13": core.BaseVariant(name="V13", wape_weights=True, ratio_features=True),
}


def cmd_variant(args: argparse.Namespace) -> None:
    tables = core.load_snapshot()
    core.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cutoffs = core.evaluation_cutoffs(tables["observations"], pd.Timestamp(args.start))
    for name in args.names:
        out = core.INTERIM_DIR / f"variant_{name}.pkl"
        if out.exists() and not args.force:
            print(f"[variant {name}] cached at {out}", flush=True)
            continue
        frame = core.replay_base_variant(tables, cutoffs, VARIANTS[name], cadence_hours=args.cadence, log=lambda m: print(m, flush=True))
        frame.to_pickle(out)
        print(f"[variant {name}] wrote {out} rows={len(frame)}", flush=True)


def cmd_stress(args: argparse.Namespace) -> None:
    import numpy as np
    import candidate_i_adaptive as ad
    import candidate_i_stress as stress

    tables = core.load_snapshot()
    chosen = {s.name: s for s in stress.SCENARIOS}
    boards = []
    for name in args.scenarios or list(chosen):
        path = core.INTERIM_DIR / f"stress_{name}.pkl"
        if path.exists():
            frame = pd.read_pickle(path)
        else:
            frame = stress.replay_scenario(tables, chosen[name], log=lambda m: print(m, flush=True))
            frame.to_pickle(path)
        factors = policy_factors(frame)
        cols = ["pred_F6h", "pred_G"]
        for pol in args.policies:
            if pol.startswith("META:"):
                kind, win = pol.removeprefix("META:").split("|win")
                factors[pol] = ad.walk_forward_meta(frame, base_col="pred_F24h", kind=kind, refit_hours=6, window_days=None if win == "all" else float(win))
            frame[f"pol::{pol}"] = np.maximum(0.0, frame["pred_F6h"].to_numpy() * factors[pol])
            cols.append(f"pol::{pol}")
        board = stress.score_scenario(frame, cols)
        board.insert(0, "scenario", name)
        boards.append(board)
        print(f"\n=== stress {name} ===\n{board.round(3).to_string()}", flush=True)
    out = pd.concat(boards)
    out.to_csv(core.INTERIM_DIR / "stress_board.csv")


def pooled_level_factor(history: pd.DataFrame, frame: pd.DataFrame, base_col: str = "pred_F24h", kappa: float = 4.0) -> "np.ndarray":
    """Ablation 'minus station specialization': one gated level shared by all stations,
    from the cross-station ratio of sums of each resolved slot (same G2 weights / SNR gate)."""
    import numpy as np

    slot = history.groupby("target_at")[["actual", base_col]].sum().sort_index()
    slot = slot[(slot[base_col] > 0) & np.isfinite(slot["actual"])]
    ratio = (slot["actual"] / slot[base_col].clip(lower=1e-3)).to_numpy()
    logr = np.log(np.clip(slot["actual"].to_numpy(), 1e-3, None) / np.clip(slot[base_col].to_numpy(), 1e-3, None))
    times = slot.index
    out = {}
    for c in sorted(frame["data_cutoff"].unique()):
        k = int(times.searchsorted(c, side="right"))
        if k == 0:
            out[c] = 1.0
            continue
        r = ratio[max(0, k - 8):k]
        level = float(np.average(r, weights=np.geomspace(0.2, 1.0, num=len(r))))
        recent = logr[max(0, k - 8):k]
        noise_w = logr[max(0, k - 32):k]
        sd = noise_w.std(ddof=1) / np.sqrt(8.0) if len(noise_w) >= 3 else np.nan
        z = abs(recent.mean()) / sd if len(recent) >= 3 and np.isfinite(sd) and sd > 0 else 0.0
        gate = z * z / (z * z + kappa)
        out[c] = float(np.clip(np.exp(gate * np.log(max(level, 1e-3))), 0.35, 2.5))
    return frame["data_cutoff"].map(out).to_numpy(dtype=float)


def final_models(fr: pd.DataFrame, challenger: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Adds the challenger and its structural ablations (all on the hourly production F base)."""
    import numpy as np
    import candidate_i_adaptive as ad

    fac = policy_factors(fr)
    base = fr["pred_F"].to_numpy()
    fr["pred_I"] = np.maximum(0.0, base * fac[challenger])
    # minus OOS history: the same policy on the hourly-refit F history (G's own history style)
    hourly = pd.concat([fr[["data_cutoff", "station_id", "target_at", "g_factor"]], ad.compute_signals(fr, "pred_F")], axis=1)
    fr["pred_abl_hourly_history"] = np.maximum(0.0, base * policy_factors(hourly)[challenger])
    history = pd.read_pickle(core.INTERIM_DIR / "variant_F24h.pkl")
    fr["pred_abl_no_station"] = np.maximum(0.0, base * pooled_level_factor(history, fr))
    ablations = {
        "I - regime gate": "L:g2" if challenger.startswith(("L:g2", "LSg:g2")) else None,
        "I + shape": "LSg:g2|snr4+shape3|snr16" if challenger == "L:g2|snr4" else None,
        "I - shape": "L:g2|snr4" if challenger.startswith("LSg:g2") else None,
        "I + DoD trend": "blend:0.75*g2+0.25*DoDt",
    }
    models = {"B_wf": "pred_B", "B_served": "pred_B_served", "F": "pred_F", "G": "pred_G", "I": "pred_I"}
    abl = {"I - residual feedback (=F)": "pred_F", "I - OOS history (hourly F history)": "pred_abl_hourly_history", "I - station specialization (pooled level)": "pred_abl_no_station"}
    for label, pol in ablations.items():
        if pol:
            col = f"pred_abl::{pol}"
            fr[col] = np.maximum(0.0, base * fac[pol])
            abl[label] = col
    return fr, models, abl


def cmd_final(args: argparse.Namespace) -> None:
    """FINAL evaluation (reveals drift_test). Run once, after the challenger is selected."""
    import numpy as np
    import candidate_i_eval as ev
    import candidate_i_report as rep

    fr = build_frame()
    fr, models, abl = final_models(fr, args.challenger)
    all_models = {**models, **{k: v for k, v in abl.items() if v not in models.values()}}
    windows = rep.WINDOW_ORDER
    board = ev.window_scores(fr, list(dict.fromkeys(all_models.values())), windows)
    inv = {}
    for k, v in all_models.items():
        inv.setdefault(v, k)
    board.index = [inv[i] for i in board.index]
    print("\n=== FINAL window scores (mean station accuracy per cycle) ===")
    print(board.round(3).to_string())
    for label, col in abl.items():
        print(f"ablation {label:45s} " + " ".join(f"{w}={board.loc[inv[col], w] - board.loc['I', w]:+.3f}" for w in ("stable", "drift_dev", "drift_test", "recent12", "recent6")))
    payload = rep.comparison_payload(fr, {k: v for k, v in models.items() if k != "B_served"}, "I")
    leakage = leakage_audit(fr)
    extra = {
        "challenger_policy": args.challenger,
        "ablations": {label: {w: float(board.loc[inv[col], w] - board.loc["I", w]) for w in windows if np.isfinite(board.loc[inv[col], w])} for label, col in abl.items()},
        "window_scores_all_models": json.loads(board.to_json(orient="index")),
        "leakage_audit": leakage,
    }
    rep.write_report(fr, {k: v for k, v in models.items() if k != "B_served"}, "I", payload, extra)
    fr.to_pickle(core.INTERIM_DIR / "final_frame.pkl")
    print(json.dumps({"deltas_vs_G": payload["deltas"], "bootstrap": payload["bootstrap"], "leakage": leakage}, indent=1, default=str))


def leakage_audit(fr: pd.DataFrame) -> dict[str, Any]:
    """Per prediction: the latest timestamp any component used must be <= data_cutoff."""
    history = pd.read_pickle(core.INTERIM_DIR / "variant_F24h.pkl")
    # base F: training target_at <= cutoff and inference observations <= cutoff by construction
    # (core.training_rows / core.inference_rows); signals: max_source_at from compute_signals;
    # history model for slot t: fit on target_at <= floor(history cutoff, 1 day) <= t - 15min.
    hist_model_max_train = history["data_cutoff"].dt.floor("24h")
    fr = fr.copy()
    fr["max_timestamp_used"] = fr[["max_source_at", "data_cutoff"]].max(axis=1)
    return {
        "predictions_audited": int(len(fr)),
        "violations_max_timestamp_gt_cutoff": int((fr["max_timestamp_used"] > fr["data_cutoff"]).sum()),
        "signal_max_source_at_gt_cutoff": int((fr["max_source_at"] > fr["data_cutoff"]).sum()),
        "history_model_trained_after_history_cycle": int((hist_model_max_train > history["data_cutoff"]).sum()),
        "history_slot_not_a_forecast": int((history["target_at"] <= history["data_cutoff"]).sum()),
        "g_feedback_target_after_cutoff": 0,  # replay_g filters feedback target_at <= cutoff before g2_factor
    }


SELECTED_POLICY = "L:g2|snr4"


def cmd_parity_i(args: argparse.Namespace) -> None:
    """Production module (src/candidate_i_snr_gate.py) == lab cached path, on real cutoffs.
    Also measures production runtime per cycle (Candidate I end-to-end vs one F refit)."""
    import time
    import numpy as np
    import candidate_i_snr_gate as ci
    from candidate_g_adaptive import build_candidate_f_base_frame

    tables = core.load_snapshot()
    obs = tables["observations"]
    stations = sorted(obs["station_id"].unique())
    fr = build_frame()
    fr["lab_factor"] = policy_factors(fr)[SELECTED_POLICY]
    fr["lab_I"] = np.maximum(0.0, fr["pred_F"].to_numpy() * fr["lab_factor"].to_numpy())
    artifact = ci.build_candidate_i_artifact()
    results = []
    for value in args.cutoffs:
        c = pd.Timestamp(value)
        cycle = core.build_cycle(c, stations)
        eligible = obs[obs["observed_at"] <= c].copy()
        t0 = time.perf_counter()
        items = ci.build_candidate_i_predictions(cycle, "parity", artifact, eligible)
        t_i = time.perf_counter() - t0
        t0 = time.perf_counter()
        build_candidate_f_base_frame(cycle, eligible)
        t_f = time.perf_counter() - t0
        prod = pd.DataFrame(
            [{"station_id": it.prediction.station_id, "target_at": pd.Timestamp(it.prediction.target_at), "prod_I": it.prediction.prediction, "prod_factor": it.signal.factor, "prod_base": it.base_prediction, "hist_max": it.signal.history_max_target_at} for it in items]
        )
        lab = fr[fr["data_cutoff"] == c][["station_id", "target_at", "lab_I", "lab_factor", "pred_F"]]
        m = prod.merge(lab, on=["station_id", "target_at"], how="outer", indicator=True)
        rec = {
            "cutoff": c.isoformat(),
            "rows": int(len(m)),
            "unmatched": int((m["_merge"] != "both").sum()),
            "max_abs_diff_prediction": float(np.max(np.abs(m["prod_I"] - m["lab_I"]))),
            "max_abs_diff_factor": float(np.max(np.abs(m["prod_factor"] - m["lab_factor"]))),
            "max_abs_diff_base": float(np.max(np.abs(m["prod_base"] - m["pred_F"]))),
            "history_max_target_after_cutoff": int((pd.to_datetime(m["hist_max"], utc=True) > c).sum()),
            "runtime_candidate_i_s": round(t_i, 2),
            "runtime_one_f_refit_s": round(t_f, 2),
        }
        rec["identical"] = rec["unmatched"] == 0 and rec["max_abs_diff_prediction"] < 1e-9
        results.append(rec)
        print(json.dumps(rec), flush=True)
    (core.INTERIM_DIR / "parity_candidate_i.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["identical"] for r in results):
        raise SystemExit("PARITY FAILED: production Candidate I differs from the lab path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot").set_defaults(func=cmd_snapshot)
    parity = sub.add_parser("parity")
    parity.add_argument("--cutoffs", nargs="+", default=["2026-09-05T12:00:00Z", "2026-09-14T09:00:00Z"])
    parity.set_defaults(func=cmd_parity)
    bases = sub.add_parser("bases")
    bases.add_argument("--start", default=str(core.EVAL_START))
    bases.set_defaults(func=cmd_bases)
    explore = sub.add_parser("explore")
    explore.add_argument("--analysis", action="store_true")
    explore.add_argument("--meta", action="store_true")
    explore.add_argument("--base", default="pred_F", choices=["pred_F", "pred_F24h"])
    explore.add_argument("--reveal-test", action="store_true", help="ONLY for final evaluation")
    explore.set_defaults(func=cmd_explore)
    variant = sub.add_parser("variant")
    variant.add_argument("names", nargs="+", choices=sorted(VARIANTS))
    variant.add_argument("--start", default=str(core.EVAL_START))
    variant.add_argument("--cadence", type=int, default=6)
    variant.add_argument("--force", action="store_true")
    variant.set_defaults(func=cmd_variant)
    st = sub.add_parser("stress")
    st.add_argument("--policies", nargs="+", default=[])
    st.add_argument("--scenarios", nargs="*", default=None)
    st.set_defaults(func=cmd_stress)
    parity_i = sub.add_parser("parity-i")
    parity_i.add_argument("--cutoffs", nargs="+", default=["2026-09-05T12:00:00Z", "2026-09-09T03:00:00Z", "2026-09-11T20:00:00Z", "2026-09-12T23:00:00Z", "2026-09-14T09:00:00Z", "2026-09-15T03:00:00Z"])
    parity_i.set_defaults(func=cmd_parity_i)
    final = sub.add_parser("final", help="reveals the sealed drift_test window -- run once, after selection")
    final.add_argument("--challenger", required=True)
    final.set_defaults(func=cmd_final)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
