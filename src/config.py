"""
Framework Configuration Loader.

Reads config.yaml from the project root and applies PIPELINE_* environment
variable overrides on top.  Env vars always take precedence so that
run_wfo.py / run_ablation.py can override specific keys per subprocess call
without touching the YAML file.

Usage
-----
    from src.config import cfg

    cfg.data.start          # "2015-01-01"
    cfg.training.lr         # 1e-3
    cfg.cvar.alpha          # 0.05

Override precedence (high → low):
    1. PIPELINE_* environment variables
    2. config.yaml values
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = _ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_ns(d: Any) -> Any:
    """Recursively convert nested dicts into SimpleNamespace objects.

    YAML's safe_load parses scientific notation like 1.0e7 as a string when
    the value cannot be decoded as a plain float (some YAML versions).  We
    coerce any string that is a valid float back to float here so downstream
    code can do arithmetic without explicit casts.
    """
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, _deep_ns(v))
        return ns
    if isinstance(d, list):
        return [_deep_ns(item) for item in d]
    if isinstance(d, str):
        try:
            return float(d)
        except ValueError:
            pass
    return d


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _apply_env_overrides(raw: dict) -> dict:
    """Apply PIPELINE_* env var overrides in-place on the raw dict copy."""
    raw = {k: (v.copy() if isinstance(v, dict) else v) for k, v in raw.items()}

    # --- data section --------------------------------------------------------
    d = raw.setdefault("data", {})
    if (v := os.environ.get("PIPELINE_DATA_START")) is not None:
        d["start"] = v
    if (v := os.environ.get("PIPELINE_DATA_END")) is not None:
        d["end"] = v
    if (v := os.environ.get("PIPELINE_INSAMPLE_END")) is not None:
        d["insample_end"] = v
    if (v := os.environ.get("PIPELINE_OOS_END")) is not None:
        d["oos_end"] = v
    # Resolve null oos_end → data.end
    if d.get("oos_end") is None:
        d["oos_end"] = d.get("end")

    # --- training section ----------------------------------------------------
    t = raw.setdefault("training", {})
    if (v := os.environ.get("PIPELINE_N_EPOCHS")) is not None:
        t["n_epochs"] = int(v)
    if (v := os.environ.get("PIPELINE_TC")) is not None:
        t["transaction_cost"] = float(v)

    # --- portfolio section ---------------------------------------------------
    p = raw.setdefault("portfolio", {})
    if (v := os.environ.get("PIPELINE_K")) is not None:
        p["K"] = int(v)
    if (v := os.environ.get("PIPELINE_TC_BPS")) is not None:
        p["transaction_cost_bps"] = float(v)

    # --- cvar section --------------------------------------------------------
    c = raw.setdefault("cvar", {})
    if (v := os.environ.get("PIPELINE_CVAR_ALPHA")) is not None:
        c["alpha"] = float(v)

    # --- paths section -------------------------------------------------------
    pth = raw.setdefault("paths", {})
    if (v := os.environ.get("PIPELINE_RESULTS_TAG")) is not None:
        pth["results_tag"] = v

    return raw


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config() -> SimpleNamespace:
    """Load config.yaml, apply env overrides, and return a SimpleNamespace tree."""
    raw = _load_yaml()
    raw = _apply_env_overrides(raw)
    return _deep_ns(raw)


# Module-level singleton.  All run_*.py scripts import this directly:
#   from src.config import cfg
cfg: SimpleNamespace = load_config()
