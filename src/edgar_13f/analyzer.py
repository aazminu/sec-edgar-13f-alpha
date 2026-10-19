"""SEC EDGAR 13F-HR institutional holdings & Form 4 insider transaction parser,
CUSIP Mod-10 validator, institutional crowding z-score calculator, and temporal
clone portfolio alpha backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# CUSIP ANSI X9.6 Mod-10 Double-Add-Double Validation & Ticker Mapping
# ---------------------------------------------------------------------------

_CUSIP_SPECIAL_VALUES: Dict[str, int] = {
    "*": 36,
    "@": 37,
    "#": 38,
}


def _cusip_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    upper = ch.upper()
    if "A" <= upper <= "Z":
        return ord(upper) - ord("A") + 10
    if upper in _CUSIP_SPECIAL_VALUES:
        return _CUSIP_SPECIAL_VALUES[upper]
    raise ValueError(f"Invalid CUSIP character: {ch!r}")


def compute_cusip_check_digit(cusip8: str) -> str:
    """Compute the 9th check digit for an 8-character CUSIP prefix using ANSI X9.6 Mod-10."""
    cleaned = cusip8.strip().upper()
    if len(cleaned) != 8:
        raise ValueError(f"Expected 8-character CUSIP body, got length {len(cleaned)}: {cusip8!r}")
    total = 0
    for idx, ch in enumerate(cleaned):
        val = _cusip_char_value(ch)
        # 1-based even positions (index 1, 3, 5, 7) are multiplied by 2
        if (idx + 1) % 2 == 0:
            val *= 2
        total += (val // 10) + (val % 10)
    check = (10 - (total % 10)) % 10
    return str(check)


def validate_cusip(cusip: str) -> bool:
    """Return True if ``cusip`` is a valid 9-character CUSIP with a matching Mod-10 check digit."""
    if not isinstance(cusip, str):
        return False
    cleaned = cusip.strip().upper()
    if len(cleaned) != 9 or not cleaned[8].isdigit():
        return False
    try:
        expected = compute_cusip_check_digit(cleaned[:8])
    except ValueError:
        return False
    return cleaned[8] == expected


DEFAULT_CUSIP_TO_TICKER: Dict[str, str] = {
    "037833100": "AAPL",
    "594918104": "MSFT",
    "67066G104": "NVDA",
    "023135106": "AMZN",
    "30303M102": "META",
    "02079K305": "GOOGL",
    "02079K107": "GOOG",
    "88160R101": "TSLA",
    "084670702": "BRK.B",
    "46625H100": "JPM",
    "92826C839": "V",
    "532457108": "LLY",
    "91324P102": "UNH",
    "007903107": "AMD",
    "64110L106": "NFLX",
}


class CusipTickerMapper:
    """Resolves 9-character CUSIPs to canonical equity tickers with issuer fallback."""

    def __init__(self, custom_map: Optional[Dict[str, str]] = None) -> None:
        self._map: Dict[str, str] = dict(DEFAULT_CUSIP_TO_TICKER)
        if custom_map:
            for k, v in custom_map.items():
                self.register(k, v)

    def register(self, cusip: str, ticker: str) -> None:
        normalized = cusip.strip().upper()
        if not validate_cusip(normalized):
            raise ValueError(f"Invalid CUSIP {cusip!r} (failed Mod-10 check digit)")
        self._map[normalized] = ticker.strip().upper()

    def resolve(self, cusip: str, issuer_name: str = "") -> str:
        normalized = cusip.strip().upper()
        if normalized in self._map:
            return self._map[normalized]
        if issuer_name:
            token = "".join(ch for ch in issuer_name.upper().split()[0] if ch.isalnum())
            if token:
                return token[:5]
        return normalized


# ---------------------------------------------------------------------------
# Data Models for 13F-HR and Form 4 Filings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldingRecord:
    """Single aggregated position entry in a Form 13F-HR Information Table."""

    cusip: str
    ticker: str
    issuer_name: str
    title_of_class: str
    value_usd: float
    shares: int
    sh_prn_type: str = "SH"
    investment_discretion: str = "SOLE"
    put_call: Optional[str] = None
    voting_sole: int = 0
    voting_shared: int = 0
    voting_none: int = 0

    @property
    def is_common_equity(self) -> bool:
        return self.put_call is None and self.sh_prn_type.upper() == "SH"


@dataclass
class Filing13F:
    """Parsed institutional Form 13F-HR quarterly filing."""

    cik: str
    manager_name: str
    period_of_report: str
    filing_date: str
    holdings: List[HoldingRecord] = field(default_factory=list)

    @property
    def total_value_usd(self) -> float:
        return sum(h.value_usd for h in self.holdings)

    @property
    def equity_value_usd(self) -> float:
        return sum(h.value_usd for h in self.holdings if h.is_common_equity)


@dataclass(frozen=True)
class InsiderTransaction:
    """Parsed SEC Form 4 non-derivative open-market or award transaction."""

    cik: str
    issuer_ticker: str
    insider_name: str
    is_officer: bool
    is_director: bool
    is_ten_pct_owner: bool
    transaction_code: str
    shares: float
    price_per_share: float
    acquired_disposed: str
    transaction_date: str
    officer_title: str = ""

    @property
    def notional_usd(self) -> float:
        return round(self.shares * self.price_per_share, 2)


@dataclass(frozen=True)
class TurnoverReport:
    """Quarter-over-quarter portfolio turnover and position changes for a 13F manager."""

    cik: str
    prev_period: str
    curr_period: str
    turnover_ratio: float
    initiated_tickers: List[str]
    exited_tickers: List[str]
    increased_tickers: List[str]
    trimmed_tickers: List[str]


@dataclass(frozen=True)
class CrowdingMetric:
    """Cross-sectional institutional ownership and crowding metric for a single security."""

    ticker: str
    holder_count: int
    ownership_breadth: float
    total_shares_held: int
    pct_of_float: Optional[float]
    avg_portfolio_weight: float
    max_portfolio_weight: float
    raw_crowding_score: float
    crowding_z_score: float


@dataclass(frozen=True)
class BacktestResult:
    """Performance attribution summary for a temporal 13F clone strategy."""

    mode: str
    periods: List[str]
    portfolio_returns: List[float]
    benchmark_returns: List[float]
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    beta: float
    jensen_alpha_annualized: float
    information_ratio: float
    max_drawdown: float


# ---------------------------------------------------------------------------
# SEC EDGAR 13F-HR XML / JSON & Form 4 XML Parsers
# ---------------------------------------------------------------------------


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child_text(elem: ET.Element, child_name: str, default: str = "") -> str:
    target = child_name.lower()
    for child in elem:
        if _strip_ns(child.tag).lower() == target:
            return (child.text or "").strip()
    return default


def _find_child(elem: ET.Element, child_name: str) -> Optional[ET.Element]:
    target = child_name.lower()
    for child in elem:
        if _strip_ns(child.tag).lower() == target:
            return child
    return None


def parse_13f_xml(
    xml_content: str,
    cik: str = "0000000000",
    manager_name: str = "UNKNOWN MANAGER",
    period_of_report: str = "2025-12-31",
    filing_date: str = "2026-02-14",
    mapper: Optional[CusipTickerMapper] = None,
    value_in_thousands: bool = False,
    validate_cusips: bool = True,
) -> Filing13F:
    """Parse an SEC EDGAR 13F-HR ``informationTable`` XML document.

    Aggregates multiple sub-manager ``<infoTable>`` entries sharing the same
    ``(cusip, putCall, shrsOrPrnAmt/sshPrnamtType)`` key into unified holdings.
    """
    if not xml_content or not xml_content.strip():
        raise ValueError("Empty 13F XML content")

    cusip_mapper = mapper or CusipTickerMapper()
    root = ET.fromstring(xml_content.strip())
    scale = 1000.0 if value_in_thousands else 1.0

    # Locate all <infoTable> elements regardless of XML namespace prefix
    info_nodes = [
        el for el in root.iter() if _strip_ns(el.tag).lower() == "infotable"
    ]
    if not info_nodes:
        raise ValueError("No <infoTable> elements found in 13F XML document")

    aggregated: Dict[Tuple[str, Optional[str], str], HoldingRecord] = {}

    for node in info_nodes:
        issuer = _find_child_text(node, "nameOfIssuer", "UNKNOWN")
        title = _find_child_text(node, "titleOfClass", "COM")
        cusip = _find_child_text(node, "cusip", "").upper()
        if validate_cusips and not validate_cusip(cusip):
            raise ValueError(f"Invalid CUSIP {cusip!r} in 13F <infoTable> for {issuer!r}")

        val_raw = float(_find_child_text(node, "value", "0") or 0.0) * scale
        shrs_node = _find_child(node, "shrsOrPrnAmt")
        shares = int(float(_find_child_text(shrs_node, "sshPrnamt", "0"))) if shrs_node is not None else 0
        sh_type = _find_child_text(shrs_node, "sshPrnamtType", "SH").upper() if shrs_node is not None else "SH"

        discretion = _find_child_text(node, "investmentDiscretion", "SOLE").upper()
        put_call_raw = _find_child_text(node, "putCall", "").upper()
        put_call = put_call_raw if put_call_raw in ("PUT", "CALL") else None

        voting_node = _find_child(node, "votingAuthority")
        v_sole = int(float(_find_child_text(voting_node, "Sole", "0"))) if voting_node is not None else 0
        v_shared = int(float(_find_child_text(voting_node, "Shared", "0"))) if voting_node is not None else 0
        v_none = int(float(_find_child_text(voting_node, "None", "0"))) if voting_node is not None else 0

        ticker = cusip_mapper.resolve(cusip, issuer)
        key = (cusip, put_call, sh_type)

        if key in aggregated:
            prev = aggregated[key]
            aggregated[key] = HoldingRecord(
                cusip=cusip,
                ticker=ticker,
                issuer_name=prev.issuer_name,
                title_of_class=prev.title_of_class,
                value_usd=round(prev.value_usd + val_raw, 2),
                shares=prev.shares + shares,
                sh_prn_type=sh_type,
                investment_discretion=prev.investment_discretion,
                put_call=put_call,
                voting_sole=prev.voting_sole + v_sole,
                voting_shared=prev.voting_shared + v_shared,
                voting_none=prev.voting_none + v_none,
            )
        else:
            aggregated[key] = HoldingRecord(
                cusip=cusip,
                ticker=ticker,
                issuer_name=issuer,
                title_of_class=title,
                value_usd=round(val_raw, 2),
                shares=shares,
                sh_prn_type=sh_type,
                investment_discretion=discretion,
                put_call=put_call,
                voting_sole=v_sole,
                voting_shared=v_shared,
                voting_none=v_none,
            )

    holdings = sorted(aggregated.values(), key=lambda h: (-h.value_usd, h.ticker))
    return Filing13F(
        cik=cik,
        manager_name=manager_name,
        period_of_report=period_of_report,
        filing_date=filing_date,
        holdings=holdings,
    )


def parse_13f_json(
    payload: str | Dict[str, Any],
    mapper: Optional[CusipTickerMapper] = None,
    validate_cusips: bool = True,
) -> Filing13F:
    """Parse a structured JSON representation of a 13F-HR filing."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    cusip_mapper = mapper or CusipTickerMapper()
    records: List[HoldingRecord] = []
    for item in data.get("holdings", []):
        cusip = str(item["cusip"]).strip().upper()
        if validate_cusips and not validate_cusip(cusip):
            raise ValueError(f"Invalid CUSIP {cusip!r} in JSON 13F payload")
        issuer = item.get("issuer_name", "UNKNOWN")
        ticker = item.get("ticker") or cusip_mapper.resolve(cusip, issuer)
        records.append(
            HoldingRecord(
                cusip=cusip,
                ticker=ticker,
                issuer_name=issuer,
                title_of_class=item.get("title_of_class", "COM"),
                value_usd=float(item["value_usd"]),
                shares=int(item["shares"]),
                sh_prn_type=item.get("sh_prn_type", "SH"),
                investment_discretion=item.get("investment_discretion", "SOLE"),
                put_call=item.get("put_call"),
                voting_sole=int(item.get("voting_sole", 0)),
                voting_shared=int(item.get("voting_shared", 0)),
                voting_none=int(item.get("voting_none", 0)),
            )
        )
    return Filing13F(
        cik=str(data.get("cik", "0000000000")),
        manager_name=str(data.get("manager_name", "UNKNOWN MANAGER")),
        period_of_report=str(data.get("period_of_report", "2025-12-31")),
        filing_date=str(data.get("filing_date", "2026-02-14")),
        holdings=records,
    )


def parse_form4_xml(xml_content: str) -> List[InsiderTransaction]:
    """Parse SEC EDGAR Form 4 ``ownershipDocument`` XML into :class:`InsiderTransaction` items."""
    root = ET.fromstring(xml_content.strip())
    issuer_node = _find_child(root, "issuer")
    cik = _find_child_text(issuer_node, "issuerCik", "") if issuer_node is not None else ""
    ticker = _find_child_text(issuer_node, "issuerTradingSymbol", "").upper() if issuer_node is not None else ""

    owner_node = _find_child(root, "reportingOwner")
    owner_id = _find_child(owner_node, "reportingOwnerId") if owner_node is not None else None
    insider_name = _find_child_text(owner_id, "rptOwnerName", "UNKNOWN") if owner_id is not None else "UNKNOWN"

    rel_node = _find_child(owner_node, "reportingOwnerRelationship") if owner_node is not None else None
    is_dir = _find_child_text(rel_node, "isDirector", "0") in ("1", "true", "True") if rel_node is not None else False
    is_off = _find_child_text(rel_node, "isOfficer", "0") in ("1", "true", "True") if rel_node is not None else False
    is_ten = (
        _find_child_text(rel_node, "isTenPercentOwner", "0") in ("1", "true", "True")
        if rel_node is not None
        else False
    )
    officer_title = _find_child_text(rel_node, "officerTitle", "") if rel_node is not None else ""

    tx_nodes = [
        el for el in root.iter() if _strip_ns(el.tag).lower() == "nonderivativetransaction"
    ]
    results: List[InsiderTransaction] = []
    for tx in tx_nodes:
        date_node = _find_child(tx, "transactionDate")
        tx_date = _find_child_text(date_node, "value", "") if date_node is not None else ""
        coding_node = _find_child(tx, "transactionCoding")
        tx_code = _find_child_text(coding_node, "transactionCode", "P").upper() if coding_node is not None else "P"

        amounts_node = _find_child(tx, "transactionAmounts")
        sh_node = _find_child(amounts_node, "transactionShares") if amounts_node is not None else None
        px_node = _find_child(amounts_node, "transactionPricePerShare") if amounts_node is not None else None
        ad_node = _find_child(amounts_node, "transactionAcquiredDisposedCode") if amounts_node is not None else None

        shares = float(_find_child_text(sh_node, "value", "0") or 0.0) if sh_node is not None else 0.0
        price = float(_find_child_text(px_node, "value", "0") or 0.0) if px_node is not None else 0.0
        acq_disp = _find_child_text(ad_node, "value", "A").upper() if ad_node is not None else "A"

        results.append(
            InsiderTransaction(
                cik=cik,
                issuer_ticker=ticker,
                insider_name=insider_name,
                is_officer=is_off,
                is_director=is_dir,
                is_ten_pct_owner=is_ten,
                transaction_code=tx_code,
                shares=shares,
                price_per_share=price,
                acquired_disposed=acq_disp,
                transaction_date=tx_date,
                officer_title=officer_title,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Portfolio Weights, Concentration (HHI), Turnover & Insider Cluster Scoring
# ---------------------------------------------------------------------------


def compute_portfolio_weights(filing: Filing13F, equity_only: bool = True) -> Dict[str, float]:
    """Compute normalized portfolio weights per ticker for a 13F filing."""
    by_ticker: Dict[str, float] = {}
    for h in filing.holdings:
        if equity_only and not h.is_common_equity:
            continue
        if h.value_usd <= 0:
            continue
        by_ticker[h.ticker] = by_ticker.get(h.ticker, 0.0) + h.value_usd

    total = sum(by_ticker.values())
    if total <= 0:
        return {}
    return {t: val / total for t, val in by_ticker.items()}


def compute_hhi_concentration(weights: Dict[str, float]) -> float:
    """Compute the Herfindahl-Hirschman Index (HHI = sum(w_i^2)) of portfolio weights."""
    if not weights:
        return 0.0
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum((w / total) ** 2 for w in weights.values())


def compute_qoq_turnover(prev_filing: Filing13F, curr_filing: Filing13F) -> TurnoverReport:
    """Compute one-way quarter-over-quarter portfolio turnover and position transitions.

    One-way turnover is defined as ``0.5 * sum_i |w_{i,t} - w_{i,t-1}|`` bounded in ``[0.0, 1.0]``.
    """
    w_prev = compute_portfolio_weights(prev_filing, equity_only=True)
    w_curr = compute_portfolio_weights(curr_filing, equity_only=True)
    all_tickers = sorted(set(w_prev.keys()) | set(w_curr.keys()))

    l1_diff = sum(abs(w_curr.get(t, 0.0) - w_prev.get(t, 0.0)) for t in all_tickers)
    turnover = round(0.5 * l1_diff, 6)

    initiated: List[str] = []
    exited: List[str] = []
    increased: List[str] = []
    trimmed: List[str] = []

    for t in all_tickers:
        p = w_prev.get(t, 0.0)
        c = w_curr.get(t, 0.0)
        if p == 0.0 and c > 0.0:
            initiated.append(t)
        elif p > 0.0 and c == 0.0:
            exited.append(t)
        elif c - p > 1e-6:
            increased.append(t)
        elif p - c > 1e-6:
            trimmed.append(t)

    return TurnoverReport(
        cik=curr_filing.cik,
        prev_period=prev_filing.period_of_report,
        curr_period=curr_filing.period_of_report,
        turnover_ratio=turnover,
        initiated_tickers=initiated,
        exited_tickers=exited,
        increased_tickers=increased,
        trimmed_tickers=trimmed,
    )


def compute_insider_cluster_score(transactions: Iterable[InsiderTransaction]) -> Dict[str, float]:
    """Compute net Form 4 insider conviction cluster score per ticker.

    Only open-market purchases (``transaction_code == 'P'``) and open-market sales
    (``transaction_code == 'S'``) are scored; option grants/awards (``A``) and tax
    withholdings (``F``) are ignored. Role multipliers weight C-suite/Officers (1.5x),
    Directors (1.2x), and 10% owners (1.0x), scaled by log notional and distinct buyer breadth.
    """
    by_ticker: Dict[str, List[InsiderTransaction]] = {}
    for tx in transactions:
        if tx.transaction_code in ("P", "S") and tx.shares > 0 and tx.price_per_share > 0:
            by_ticker.setdefault(tx.issuer_ticker, []).append(tx)

    scores: Dict[str, float] = {}
    for ticker, txs in by_ticker.items():
        distinct_buyers = {t.insider_name for t in txs if t.transaction_code == "P"}
        distinct_sellers = {t.insider_name for t in txs if t.transaction_code == "S"}
        raw = 0.0
        for t in txs:
            role_weight = 1.5 if t.is_officer else (1.2 if t.is_director else 1.0)
            sign = 1.0 if t.transaction_code == "P" else -1.0
            raw += sign * role_weight * math.log1p(t.notional_usd)

        # Cluster breadth multiplier rewards >=2 distinct open-market insider buyers
        cluster_mult = 1.0 + 0.25 * max(0, len(distinct_buyers) - 1) - 0.15 * max(0, len(distinct_sellers) - 1)
        scores[ticker] = round(raw * max(cluster_mult, 0.25), 4)

    return scores


# ---------------------------------------------------------------------------
# Institutional Crowding Z-Score & Temporal Clone Alpha Backtester
# ---------------------------------------------------------------------------


def compute_crowding_scores(
    filings: List[Filing13F],
    float_shares_by_ticker: Optional[Dict[str, int]] = None,
) -> Dict[str, CrowdingMetric]:
    """Compute cross-sectional institutional crowding z-scores across a universe of 13F filings.

    Combines ownership breadth (fraction of funds holding the stock), average portfolio
    conviction weight, and percentage of float held (when ``float_shares_by_ticker`` is
    supplied) into a composite crowding score, then standardizes into cross-sectional z-scores.
    """
    if not filings:
        return {}

    num_funds = len(filings)
    fund_weights = [compute_portfolio_weights(f, equity_only=True) for f in filings]

    shares_by_ticker: Dict[str, int] = {}
    weights_by_ticker: Dict[str, List[float]] = {}

    for filing, w_map in zip(filings, fund_weights):
        for h in filing.holdings:
            if not h.is_common_equity or h.value_usd <= 0:
                continue
            shares_by_ticker[h.ticker] = shares_by_ticker.get(h.ticker, 0) + h.shares
        for ticker, weight in w_map.items():
            weights_by_ticker.setdefault(ticker, []).append(weight)

    raw_metrics: Dict[str, Tuple[int, float, int, Optional[float], float, float, float]] = {}
    for ticker, w_list in weights_by_ticker.items():
        holder_count = len(w_list)
        breadth = holder_count / num_funds
        total_sh = shares_by_ticker.get(ticker, 0)
        avg_w = sum(w_list) / num_funds
        max_w = max(w_list)
        pct_float: Optional[float] = None
        if float_shares_by_ticker and float_shares_by_ticker.get(ticker, 0) > 0:
            pct_float = min(1.0, total_sh / float_shares_by_ticker[ticker])
            raw_score = 0.45 * breadth + 0.35 * avg_w + 0.20 * pct_float
        else:
            raw_score = 0.60 * breadth + 0.40 * avg_w

        raw_metrics[ticker] = (holder_count, breadth, total_sh, pct_float, avg_w, max_w, raw_score)

    raw_vals = [m[6] for m in raw_metrics.values()]
    mean_raw = sum(raw_vals) / len(raw_vals)
    variance = sum((v - mean_raw) ** 2 for v in raw_vals) / len(raw_vals)
    std_raw = math.sqrt(variance)

    output: Dict[str, CrowdingMetric] = {}
    for ticker, (hc, breadth, tot_sh, pct_fl, avg_w, max_w, r_score) in sorted(raw_metrics.items()):
        z_score = (r_score - mean_raw) / std_raw if std_raw > 1e-12 else 0.0
        output[ticker] = CrowdingMetric(
            ticker=ticker,
            holder_count=hc,
            ownership_breadth=round(breadth, 6),
            total_shares_held=tot_sh,
            pct_of_float=round(pct_fl, 6) if pct_fl is not None else None,
            avg_portfolio_weight=round(avg_w, 6),
            max_portfolio_weight=round(max_w, 6),
            raw_crowding_score=round(r_score, 6),
            crowding_z_score=round(z_score, 6),
        )
    return output


def construct_clone_portfolio(
    filings: List[Filing13F],
    mode: str = "conviction",
    top_n: int = 10,
    float_shares_by_ticker: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Construct a systematic 13F clone portfolio across multiple institutional managers.

    Supported ``mode`` values:
    - ``"equal"``: Top ``top_n`` most widely held equities weighted equally (1/N).
    - ``"conviction"``: Top ``top_n`` equities ranked and weighted by aggregate conviction
      (average portfolio weight across filing managers).
    - ``"anti_crowded_conviction"``: Ranks by conviction penalized by high crowding z-score
      (``avg_portfolio_weight * exp(-0.5 * max(0, crowding_z_score))``).
    """
    if mode not in ("equal", "conviction", "anti_crowded_conviction"):
        raise ValueError(f"Unsupported clone portfolio mode: {mode!r}")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    crowding = compute_crowding_scores(filings, float_shares_by_ticker=float_shares_by_ticker)
    if not crowding:
        return {}

    if mode == "equal":
        ranked = sorted(
            crowding.values(),
            key=lambda m: (m.holder_count, m.avg_portfolio_weight, m.ticker),
            reverse=True,
        )[:top_n]
        w = 1.0 / len(ranked)
        return {m.ticker: round(w, 6) for m in ranked}

    if mode == "conviction":
        ranked = sorted(
            crowding.values(),
            key=lambda m: (m.avg_portfolio_weight, m.holder_count, m.ticker),
            reverse=True,
        )[:top_n]
        total_score = sum(m.avg_portfolio_weight for m in ranked)
        return {m.ticker: m.avg_portfolio_weight / total_score for m in ranked}

    # mode == "anti_crowded_conviction"
    scored: List[Tuple[str, float]] = []
    for m in crowding.values():
        penalty = math.exp(-0.5 * max(0.0, m.crowding_z_score))
        scored.append((m.ticker, m.avg_portfolio_weight * penalty))
    scored.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)
    selected = scored[:top_n]
    total_score = sum(s for _, s in selected)
    return {t: s / total_score for t, s in selected}


def backtest_13f_strategy(
    quarterly_filings: Dict[str, List[Filing13F]],
    post_filing_period_returns: Dict[str, Dict[str, float]],
    benchmark_returns: Dict[str, float],
    mode: str = "conviction",
    top_n: int = 10,
    risk_free_rate_annual: float = 0.04,
    periods_per_year: int = 4,
) -> BacktestResult:
    """Run a point-in-time quarterly backtest of a 13F clone strategy.

    ``post_filing_period_returns[period]`` represents the realized security returns over the
    holding window *after* the 45-day statutory SEC Form 13F filing deadline for ``period``,
    preventing look-ahead bias.
    """
    periods = sorted(quarterly_filings.keys())
    if not periods:
        raise ValueError("quarterly_filings cannot be empty")

    port_rets: List[float] = []
    bench_rets: List[float] = []

    for period in periods:
        filings = quarterly_filings[period]
        weights = construct_clone_portfolio(filings, mode=mode, top_n=top_n)
        sec_rets = post_filing_period_returns.get(period, {})
        p_ret = sum(w * sec_rets.get(ticker, 0.0) for ticker, w in weights.items())
        b_ret = benchmark_returns.get(period, 0.0)
        port_rets.append(p_ret)
        bench_rets.append(b_ret)

    n = len(port_rets)
    cum_growth = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in port_rets:
        cum_growth *= 1.0 + r
        if cum_growth > peak:
            peak = cum_growth
        dd = (peak - cum_growth) / peak
        if dd > max_dd:
            max_dd = dd

    cum_return = cum_growth - 1.0
    ann_return = cum_growth ** (periods_per_year / n) - 1.0

    rf_per_period = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess_rets = [r - rf_per_period for r in port_rets]
    mean_port = sum(port_rets) / n
    mean_bench = sum(bench_rets) / n
    mean_excess = sum(excess_rets) / n

    if n > 1:
        var_port = sum((r - mean_port) ** 2 for r in port_rets) / (n - 1)
        var_bench = sum((b - mean_bench) ** 2 for b in bench_rets) / (n - 1)
        cov_pb = sum((r - mean_port) * (b - mean_bench) for r, b in zip(port_rets, bench_rets)) / (n - 1)
    else:
        var_port = 0.0
        var_bench = 0.0
        cov_pb = 0.0

    ann_vol = math.sqrt(var_port) * math.sqrt(periods_per_year)
    sharpe = (mean_excess * periods_per_year) / ann_vol if ann_vol > 1e-12 else 0.0
    beta = cov_pb / var_bench if var_bench > 1e-12 else 1.0

    # Jensen's alpha per period annualized
    alpha_per_period = (mean_port - rf_per_period) - beta * (mean_bench - rf_per_period)
    jensen_alpha_ann = alpha_per_period * periods_per_year

    active_rets = [r - b for r, b in zip(port_rets, bench_rets)]
    mean_active = sum(active_rets) / n
    if n > 1:
        var_active = sum((a - mean_active) ** 2 for a in active_rets) / (n - 1)
    else:
        var_active = 0.0
    tracking_error = math.sqrt(var_active) * math.sqrt(periods_per_year)
    info_ratio = (mean_active * periods_per_year) / tracking_error if tracking_error > 1e-12 else 0.0

    return BacktestResult(
        mode=mode,
        periods=periods,
        portfolio_returns=[round(r, 6) for r in port_rets],
        benchmark_returns=[round(b, 6) for b in bench_rets],
        cumulative_return=round(cum_return, 6),
        annualized_return=round(ann_return, 6),
        annualized_volatility=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 6),
        beta=round(beta, 6),
        jensen_alpha_annualized=round(jensen_alpha_ann, 6),
        information_ratio=round(info_ratio, 6),
        max_drawdown=round(max_dd, 6),
    )
