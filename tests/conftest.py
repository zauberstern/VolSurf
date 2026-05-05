"""
Shared pytest fixtures.

The ``wrds_data`` session fixture attempts to load a real 2-year WRDS panel
when WRDS_USERNAME and WRDS_PASSWORD are present in the environment (via
.env).  Tests decorated with ``@pytest.mark.wrds`` are automatically skipped
when the credentials are absent, so the existing synthetic-data unit suite
runs unchanged in CI or offline environments.
"""

from __future__ import annotations

import logging
import os

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _wrds_available() -> bool:
    return bool(os.environ.get("WRDS_USERNAME") and os.environ.get("WRDS_PASSWORD"))


def pytest_collection_modifyitems(config, items):
    if not _wrds_available():
        skip_wrds = pytest.mark.skip(reason="WRDS credentials not found in .env")
        for item in items:
            if item.get_closest_marker("wrds"):
                item.add_marker(skip_wrds)


@pytest.fixture(scope="session", autouse=True)
def _live_logging():
    """Stream INFO-level logs to stdout so WRDS progress is visible in real time."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


@pytest.fixture(scope="session")
def wrds_data(_live_logging) -> dict:
    """
    Session-scoped fixture returning a real 2-year WRDS panel as a dict:

        ``"prices"``       — pd.Series  (SPX closing level)
        ``"option_panel"`` — pd.DataFrame (date, iv_30, iv_91, skew_25d)
        ``"vix"``          — pd.Series  (decimal)
        ``"treasury_10y"`` — pd.Series  (decimal)

    The date window is fixed at 2023-01-01 → 2024-12-31 — the most recent
    2-year period fully covered by all four WRDS datasets (crsp.dsi data
    currency currently ends at 2024-12-31).  Using explicit dates also
    produces a stable parquet cache key so repeated test runs load from disk.

    Automatically skipped when WRDS credentials are absent.
    """
    if not _wrds_available():
        pytest.skip("WRDS credentials not found in .env")

    from src.econometrics.wrds_loader import load_wrds_data
    return load_wrds_data(start="2023-01-01", end="2024-12-31")
