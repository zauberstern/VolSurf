#!/usr/bin/env python
"""
Walk-Forward Out-of-Sample Validation — Phase 2 (ROADMAP.md)
=============================================================
Evaluates the policy over 6 expanding annual windows:

    Window | IS period        | OOS test year
    -------|------------------|---------------
    w1     | 2003 – 2018      | 2019
    w2     | 2003 – 2019      | 2020
    w3     | 2003 – 2020      | 2021
    w4     | 2003 – 2021      | 2022  ← primary published result
    w5     | 2003 – 2022      | 2023
    w6     | 2003 – 2023      | 2024

Each window re-trains from scratch on the full IS history up to
INSAMPLE_END, then evaluates on the single OOS year.  The underlying WRDS
data cache (2003–2024) is reused across all windows — only INSAMPLE_END and
PIPELINE_OOS_END differ.

The VECM simulation path cache is keyed by (N_SIM_PATHS, N_SIM_STEPS,
state_dim, T_is, is_start_date).  Since IS length differs per window, each
window generates its own path cache on first run.

Usage
-----
    python run_wfo.py                       # all 6 windows, 50 epochs each
    PIPELINE_N_EPOCHS=10 python run_wfo.py  # quick 10-epoch check
    python run_wfo.py --window w4           # single window by label

Success criteria:
  - Sharpe > 0 in at least 4 / 6 windows
  - DSR > 0 in at least 3 / 6 windows
  - No single window dominates the full-period Sharpe by >3×
    (would indicate regime over-fit to that year)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg

RESULTS_DIR = ROOT / cfg.paths.cache_dir

# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL INTEGRITY GUARD — MANDATORY READ BEFORE MODIFYING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
#
# Walk-forward evaluation is only causally valid if ALL hyperparameters in
# config.yaml were fixed BEFORE any WFO window results were observed.
#
# The WFO windows (w1: OOS=2019, w2: OOS=2020, ...) produce correlated Sharpe
# ratios.  Selecting hyperparameters (ppo_clip_eps, entropy_coef, cvar.c_bar,
# gating.alpha, model.hidden_size, etc.) by evaluating WFO performance and
# then back-reporting those same WFO results as "out-of-sample validation" is
# a form of indirect look-ahead bias sometimes called "hyperparameter leakage."
#
# This framework enforces the following protocol:
#   1.  Hyperparameters are selected ONCE using ONLY the primary IS window
#       (2003–2021) via ablation sweep (run_ablation.py).  WFO window results
#       are never used for tuning decisions.
#   2.  After hyperparameters are frozen in config.yaml, run_wfo.py is executed
#       exactly ONCE as a post-hoc robustness check.
#   3.  The earliest WFO OOS year (2019, window w1) contains the 2020 COVID
#       liquidity shock in the w2 window.  Any hyperparameter tuned using
#       information beyond 2018 contaminates the entire WFO table.
#
# Assertion: the hyperparameter freeze date is stored as cfg.wfo.hp_freeze_date.
# If it is set and falls AFTER the earliest window's IS end date, abort with a
# clear error so the contamination is never silently published.
# ══════════════════════════════════════════════════════════════════════════════
_HP_FREEZE = getattr(cfg.wfo, "hp_freeze_date", None)
_EARLIEST_IS_END = WINDOWS[0][0] if WINDOWS else None
if _HP_FREEZE and _EARLIEST_IS_END:
    import datetime as _dt
    _freeze_dt = _dt.date.fromisoformat(_HP_FREEZE)
    _earliest_dt = _dt.date.fromisoformat(_EARLIEST_IS_END)
    if _freeze_dt > _earliest_dt:
        raise RuntimeError(
            f"CAUSAL INTEGRITY VIOLATION: hyperparameters were frozen on "
            f"{_HP_FREEZE}, which is AFTER the earliest WFO IS-end date "
            f"({_EARLIEST_IS_END}).  This means WFO OOS results (year "
            f"{int(_EARLIEST_IS_END[:4]) + 1}) may have influenced the "
            f"hyperparameter selection, invalidating the WFO table.  "
            f"Freeze hyperparameters using only pre-{_EARLIEST_IS_END[:4]} "
            f"information (run_ablation.py on IS-only data) and update "
            f"wfo.hp_freeze_date in config.yaml."
        )

# ── Window schedule sourced from config.yaml (wfo.windows) ────────────────────
WINDOWS: list[tuple[str, str, str]] = [
    (w.insample_end, w.oos_end, w.label)
    for w in cfg.wfo.windows
]


def run_window(is_end: str, oos_end: str, label: str) -> dict:
    """Run one WFO window and return its metrics dict."""
    tag       = f"_wfo_{label}"
    json_path = RESULTS_DIR / f"rl_oos_results{tag}.json"

    if json_path.exists():
        print(f"\n  [SKIP]  {label}  IS→{is_end}  OOS→{oos_end}  — cache hit")
    else:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  [RUN]   {label}  IS end={is_end}  OOS={oos_end[:4]}")
        print(sep, flush=True)

        env = os.environ.copy()
        env["PIPELINE_INSAMPLE_END"]  = is_end
        env["PIPELINE_OOS_END"]       = oos_end
        env["PIPELINE_RESULTS_TAG"]   = tag
        # DATA_START / DATA_END / PIPELINE_TC / PIPELINE_CVAR_ALPHA intentionally
        # NOT overridden here so the defaults (2003-01-01, 2024-12-31, 1e-4, 0.05)
        # are used — the existing WRDS parquet caches are valid for all windows.

        result = subprocess.run(
            [sys.executable, str(ROOT / "run_pipeline.py")],
            env=env,
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"  [ERROR] pipeline exited with code {result.returncode}")
            return {"label": label, "is_end": is_end, "oos_end": oos_end, "error": True}

    if not json_path.exists():
        print(f"  [ERROR] expected JSON not found: {json_path}")
        return {"label": label, "is_end": is_end, "oos_end": oos_end, "error": True}

    raw = json.loads(json_path.read_text())
    return {
        "label":         label,
        "is_end":        is_end,
        "oos_year":      oos_end[:4],
        "final_reward":  float(raw["rewards"][-1]) if raw.get("rewards") else None,
        "dsr":           raw.get("dsr"),
        "profit_factor": raw.get("profit_factor"),
        "wfe":           raw.get("wfe"),
        "mc_pvalue":     raw.get("mc_pvalue"),
    }


def print_summary(results: list[dict]) -> None:
    sep = "=" * 80
    print(f"\n\n{sep}")
    print("  WALK-FORWARD SUMMARY")
    print(sep)
    print(f"  {'Window':>8}  {'IS End':>12}  {'OOS Yr':>7}  {'FinalRew':>10}  {'DSR':>8}  {'WFE':>8}  {'MCp':>8}")
    print("-" * 80)

    dsr_positive = 0
    for r in results:
        if r.get("error"):
            print(f"  {r['label']:>8}  {'ERROR':>40}")
            continue
        fr  = r.get("final_reward") or 0.0
        dsr = r.get("dsr") or 0.0
        wfe = r.get("wfe") or 0.0
        mc  = r.get("mc_pvalue") or 0.0
        flag = "  ← PRIMARY" if r["label"] == "w4_2022" else ""
        if dsr > 0:
            dsr_positive += 1
        print(
            f"  {r['label']:>8}  {r['is_end']:>12}  {r['oos_year']:>7}"
            f"  {fr:>+10.6f}  {dsr:>8.4f}  {wfe:>8.4f}  {mc:>8.4f}{flag}"
        )

    completed = [r for r in results if not r.get("error")]
    print(sep)
    if completed:
        print(f"\n  DSR > 0 in {dsr_positive} / {len(completed)} completed windows.")
        if dsr_positive >= 3:
            print("  ✓ Robustness criterion met (DSR > 0 in ≥ 3/6 windows).")
        else:
            print("  ✗ Robustness criterion NOT met (< 3/6 windows positive DSR).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS validation")
    parser.add_argument(
        "--window", default=None,
        help="Run a single window by label (e.g. w4_2022).  Omit to run all.",
    )
    args = parser.parse_args()

    sep = "=" * 70
    print(sep)
    print("  WALK-FORWARD OOS VALIDATION  (Phase 2 of ROADMAP.md)")
    windows_to_run = (
        [w for w in WINDOWS if w[2] == args.window]
        if args.window else WINDOWS
    )
    if not windows_to_run:
        print(f"  ERROR: unknown window label '{args.window}'")
        print(f"  Valid labels: {[w[2] for w in WINDOWS]}")
        sys.exit(1)
    print(f"  Running {len(windows_to_run)} window(s)")
    n_epochs = os.environ.get("PIPELINE_N_EPOCHS", "50")
    print(f"  Epochs per window: {n_epochs}  (set PIPELINE_N_EPOCHS to override)")
    print(sep, flush=True)

    results: list[dict] = []
    for is_end, oos_end, label in windows_to_run:
        r = run_window(is_end, oos_end, label)
        results.append(r)
        status = "ERROR" if r.get("error") else f"DSR={r.get('dsr', 0) or 0:.4f}"
        print(f"  → {label}: {status}", flush=True)

    print_summary(results)

    summary_path = RESULTS_DIR / "wfo_summary.json"
    # Merge with existing summary if running a single window
    if args.window and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing = [e for e in existing if e.get("label") != args.window]
        results  = existing + results
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"  Full summary → {summary_path}\n")


if __name__ == "__main__":
    main()
