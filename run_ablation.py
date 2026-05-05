#!/usr/bin/env python
"""
Friction & CVaR Ablation Sweep — Phase 1 (ROADMAP.md)
======================================================
Runs run_pipeline.py across a 3 × 4 parameter grid:

    CVAR_ALPHA  ∈ {1.0, 0.25, 0.05}   (1.0 = CVaR constraint disabled)
    TC          ∈ {0.0, 1e-5, 1e-4, 1e-3}

Each cell runs the full pipeline (Phases I–IV) and writes a tagged
results JSON to data/wrds_cache/.  Already-completed cells are skipped
(cache-hit logic).  A summary table and aggregate JSON are written at the
end.

Usage
-----
    python run_ablation.py                      # full 50-epoch runs
    PIPELINE_N_EPOCHS=10 python run_ablation.py # quick 10-epoch pre-check

Key success criteria (Phase 1 gate):
  - (CVAR_ALPHA=1.0, TC=0.0): mean_action > 0.01 and dsr > 0
    → alpha signal confirmed; proceed to Phase 3.
  - Smooth TC decay of mean_action (no cliff-drop to 0 at first non-zero TC)
    → Whalley-Wilmott behaviour; policy has genuine convexity to trade.
"""

from __future__ import annotations

import itertools
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

# ── Parameter grid (sourced from config.yaml ablation section) ───────────────
CVAR_ALPHAS = cfg.ablation.cvar_alphas
TC_VALUES   = cfg.ablation.tc_values
# Full grid = len(CVAR_ALPHAS) × len(TC_VALUES) cells.


def _tag(cvar_alpha: float, tc: float) -> str:
    """Build a filesystem-safe tag for this (alpha, tc) cell."""
    tc_str = f"{tc:.0e}".replace("-", "m").replace("+", "")
    alpha_str = f"{cvar_alpha:.2f}".replace(".", "p")
    return f"_abl_alpha{alpha_str}_tc{tc_str}"


def run_cell(cvar_alpha: float, tc: float) -> dict:
    """Run one pipeline cell and return its metrics dict."""
    tag       = _tag(cvar_alpha, tc)
    json_path = RESULTS_DIR / f"rl_oos_results{tag}.json"

    if json_path.exists():
        print(f"\n  [SKIP]  alpha={cvar_alpha}  tc={tc:.0e}  — cache hit ({tag})")
    else:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  [RUN]   alpha={cvar_alpha}  tc={tc:.0e}  tag={tag}")
        print(sep, flush=True)

        env = os.environ.copy()
        env["PIPELINE_CVAR_ALPHA"]  = str(cvar_alpha)
        env["PIPELINE_TC"]          = str(tc)
        env["PIPELINE_RESULTS_TAG"] = tag

        result = subprocess.run(
            [sys.executable, str(ROOT / "run_pipeline.py")],
            env=env,
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"  [ERROR] pipeline exited with code {result.returncode}")
            return {"cvar_alpha": cvar_alpha, "tc": tc, "error": True}

    if not json_path.exists():
        print(f"  [ERROR] expected JSON not found: {json_path}")
        return {"cvar_alpha": cvar_alpha, "tc": tc, "error": True}

    raw = json.loads(json_path.read_text())
    return {
        "cvar_alpha":    cvar_alpha,
        "tc":            tc,
        "final_reward":  float(raw["rewards"][-1]) if raw.get("rewards") else None,
        "mean_reward":   float(sum(raw["rewards"]) / len(raw["rewards"])) if raw.get("rewards") else None,
        "dsr":           raw.get("dsr"),
        "profit_factor": raw.get("profit_factor"),
        "wfe":           raw.get("wfe"),
        "mc_pvalue":     raw.get("mc_pvalue"),
        "final_eta":     float(raw["eta"][-1]) if raw.get("eta") else None,
    }


def print_summary(results: list[dict]) -> None:
    sep = "=" * 86
    print(f"\n\n{sep}")
    print("  ABLATION SUMMARY")
    print(sep)
    print(f"  {'CVAR_α':>8}  {'TC':>8}  {'FinalReward':>13}  {'DSR':>8}  {'ProfitF':>8}  {'MCp':>8}  {'Eta':>8}")
    print("-" * 86)
    for r in results:
        if r.get("error"):
            print(f"  {r['cvar_alpha']:>8.2f}  {r['tc']:>8.1e}  {'ERROR':>13}")
            continue
        fr   = r.get("final_reward") or 0.0
        dsr  = r.get("dsr") or 0.0
        pf   = r.get("profit_factor") or 0.0
        mc   = r.get("mc_pvalue") or 0.0
        eta  = r.get("final_eta") or 0.0
        flag = " ← GATE" if (r["cvar_alpha"] == 1.0 and r["tc"] == 0.0) else ""
        print(
            f"  {r['cvar_alpha']:>8.2f}  {r['tc']:>8.1e}"
            f"  {fr:>+13.6f}  {dsr:>8.4f}  {pf:>8.4f}  {mc:>8.4f}  {eta:>8.4f}{flag}"
        )
    print(sep)

    # Gate assessment
    gate_cell = next(
        (r for r in results if r.get("cvar_alpha") == 1.0 and r.get("tc") == 0.0),
        None,
    )
    if gate_cell and not gate_cell.get("error"):
        gate_dsr = gate_cell.get("dsr") or 0.0
        if gate_dsr > 0:
            print("\n  ✓ GATE PASSED: frictionless DSR > 0 — alpha signal confirmed.")
            print("    → Phase 3 (per-step GAE architecture) is unblocked.")
        else:
            print("\n  ✗ GATE FAILED: frictionless DSR ≤ 0 — no exploitable signal found.")
            print("    → Do NOT proceed to Phase 3; investigate feature construction.")
    else:
        print("\n  ? Gate cell (alpha=1.0, TC=0.0) not yet completed.")
    print()


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("  FRICTION & CVaR ABLATION SWEEP  (Phase 1 of ROADMAP.md)")
    print(f"  Grid: {len(CVAR_ALPHAS)} × {len(TC_VALUES)} = {len(CVAR_ALPHAS) * len(TC_VALUES)} cells")
    n_epochs = os.environ.get("PIPELINE_N_EPOCHS", "50")
    print(f"  Epochs per cell: {n_epochs}  (set PIPELINE_N_EPOCHS to override)")
    print(sep, flush=True)

    results: list[dict] = []
    for cvar_alpha, tc in itertools.product(CVAR_ALPHAS, TC_VALUES):
        r = run_cell(cvar_alpha, tc)
        results.append(r)

    print_summary(results)

    summary_path = RESULTS_DIR / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"  Full summary → {summary_path}\n")


if __name__ == "__main__":
    main()
