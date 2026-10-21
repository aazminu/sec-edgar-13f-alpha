"""SEC EDGAR 13F-HR Institutional Holdings & Form 4 Alpha Engine."""

from .analyzer import (
    BacktestResult,
    CrowdingMetric,
    CusipTickerMapper,
    Filing13F,
    HoldingRecord,
    InsiderTransaction,
    TurnoverReport,
    backtest_13f_strategy,
    compute_crowding_scores,
    compute_cusip_check_digit,
    compute_hhi_concentration,
    compute_insider_cluster_score,
    compute_portfolio_weights,
    compute_qoq_turnover,
    construct_clone_portfolio,
    parse_13f_json,
    parse_13f_xml,
    parse_form4_xml,
    validate_cusip,
)

__all__ = [
    "BacktestResult",
    "CrowdingMetric",
    "CusipTickerMapper",
    "Filing13F",
    "HoldingRecord",
    "InsiderTransaction",
    "TurnoverReport",
    "backtest_13f_strategy",
    "compute_crowding_scores",
    "compute_cusip_check_digit",
    "compute_hhi_concentration",
    "compute_insider_cluster_score",
    "compute_portfolio_weights",
    "compute_qoq_turnover",
    "construct_clone_portfolio",
    "parse_13f_json",
    "parse_13f_xml",
    "parse_form4_xml",
    "validate_cusip",
]

__version__ = "0.2.0"

from .diagnostics import run_diagnostic_suite, validate_runtime_invariants
__all__ = list(__all__) + ["run_diagnostic_suite", "validate_runtime_invariants"]
