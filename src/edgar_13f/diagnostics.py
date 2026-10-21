"""Deterministic runtime diagnostics and invariant verification for `edgar_13f`."""

from __future__ import annotations

import time
from typing import Any

from .analyzer import validate_cusip, compute_hhi_concentration

def _domain_probe(seed: int) -> dict:
    apple_valid = validate_cusip("037833100")
    msft_valid = validate_cusip("594918104")
    hhi = compute_hhi_concentration({"AAPL": 0.50, "MSFT": 0.30, "NVDA": 0.20})
    return {
        "apple_cusip_valid": apple_valid,
        "msft_cusip_valid": msft_valid,
        "hhi_concentration": round(hhi, 4),
        "invariants_ok": (apple_valid is True and msft_valid is True and abs(hhi - 0.38) < 1e-6),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `sec-edgar-13f-alpha` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "sec-edgar-13f-alpha",
        "package": "edgar_13f",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `edgar_13f`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "sec-edgar-13f-alpha",
        "package": "edgar_13f",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
