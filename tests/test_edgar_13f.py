"""Comprehensive pytest test suite for sec-edgar-13f-alpha."""

import pytest

from edgar_13f import (
    CusipTickerMapper,
    Filing13F,
    HoldingRecord,
    InsiderTransaction,
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


SAMPLE_13F_XML_NAMESPACED = """<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>45000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>200000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>200000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>15000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>66000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
    <votingAuthority>
      <Sole>50000</Sole>
      <Shared>16000</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>40000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>300000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>300000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>PUT</titleOfClass>
    <cusip>67066G104</cusip>
    <value>5000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>50000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall>Put</putCall>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>50000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
</informationTable>
"""


SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0001045810</issuerCik>
    <issuerTradingSymbol>NVDA</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>EVP &amp; CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-02-20</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>130.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_cusip_mod10_check_digit_standard_equities():
    assert compute_cusip_check_digit("03783310") == "0"  # AAPL -> 037833100
    assert compute_cusip_check_digit("59491810") == "4"  # MSFT -> 594918104
    assert compute_cusip_check_digit("67066G10") == "4"  # NVDA -> 67066G104
    assert compute_cusip_check_digit("02079K30") == "5"  # GOOGL -> 02079K305
    assert compute_cusip_check_digit("88160R10") == "1"  # TSLA -> 88160R101


def test_validate_cusip_accepts_valid_and_rejects_corrupted():
    assert validate_cusip("037833100") is True
    assert validate_cusip("67066G104") is True
    assert validate_cusip("037833109") is False
    assert validate_cusip("SHORT") is False
    assert validate_cusip(123456789) is False  # type: ignore[arg-type]


def test_cusip_special_characters_and_invalid_char():
    # Test *, @, # special characters allowed by ANSI X9.6
    chk = compute_cusip_check_digit("12345*@#")
    assert validate_cusip(f"12345*@#{chk}") is True
    with pytest.raises(ValueError, match="Invalid CUSIP character"):
        compute_cusip_check_digit("12345$10")


def test_cusip_ticker_mapper_registration_and_fallback():
    mapper = CusipTickerMapper()
    assert mapper.resolve("037833100") == "AAPL"
    assert mapper.resolve("999999999", issuer_name="PALANTIR TECHNOLOGIES INC") == "PALAN"

    # Register valid custom CUSIP
    chk = compute_cusip_check_digit("70109410")
    valid_custom = f"70109410{chk}"
    mapper.register(valid_custom, "ph")
    assert mapper.resolve(valid_custom) == "PH"

    with pytest.raises(ValueError, match="Invalid CUSIP"):
        mapper.register("037833109", "BAD")


def test_parse_13f_xml_aggregates_sub_managers_and_separates_options():
    filing = parse_13f_xml(
        SAMPLE_13F_XML_NAMESPACED,
        cik="0001067983",
        manager_name="BERKSHIRE QUANT LP",
        period_of_report="2025-12-31",
        filing_date="2026-02-14",
    )
    assert filing.total_value_usd == pytest.approx(105_000_000.0)
    assert filing.equity_value_usd == pytest.approx(100_000_000.0)
    # 2 AAPL common lines merged into 1 + 1 NVDA common + 1 NVDA PUT = 3 records
    assert len(filing.holdings) == 3

    aapl = next(h for h in filing.holdings if h.ticker == "AAPL")
    assert aapl.value_usd == pytest.approx(60_000_000.0)
    assert aapl.shares == 266_000
    assert aapl.voting_sole == 250_000
    assert aapl.voting_shared == 16_000


def test_parse_13f_xml_rejects_invalid_cusip_and_empty_input():
    with pytest.raises(ValueError, match="Empty 13F XML"):
        parse_13f_xml("  ")

    bad_xml = SAMPLE_13F_XML_NAMESPACED.replace("037833100", "037833109")
    with pytest.raises(ValueError, match="Invalid CUSIP"):
        parse_13f_xml(bad_xml)


def test_parse_13f_json_roundtrip():
    payload = {
        "cik": "0001336528",
        "manager_name": "PERSHING ALPHA FUND",
        "period_of_report": "2025-12-31",
        "filing_date": "2026-02-14",
        "holdings": [
            {"cusip": "594918104", "issuer_name": "MICROSOFT CORP", "value_usd": 80_000_000, "shares": 200_000},
            {"cusip": "02079K305", "issuer_name": "ALPHABET INC", "value_usd": 20_000_000, "shares": 110_000},
        ],
    }
    filing = parse_13f_json(payload)
    weights = compute_portfolio_weights(filing)
    assert weights["MSFT"] == pytest.approx(0.80)
    assert weights["GOOGL"] == pytest.approx(0.20)


def test_compute_portfolio_weights_excludes_options_by_default():
    filing = parse_13f_xml(SAMPLE_13F_XML_NAMESPACED)
    weights_eq = compute_portfolio_weights(filing, equity_only=True)
    assert weights_eq["AAPL"] == pytest.approx(0.60)
    assert weights_eq["NVDA"] == pytest.approx(0.40)

    weights_all = compute_portfolio_weights(filing, equity_only=False)
    assert weights_all["NVDA"] == pytest.approx(45_000_000.0 / 105_000_000.0)


def test_compute_hhi_concentration():
    # Equal 4-stock portfolio -> HHI = 4 * (0.25^2) = 0.25
    assert compute_hhi_concentration({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == pytest.approx(0.25)
    # Single-stock portfolio -> HHI = 1.0
    assert compute_hhi_concentration({"A": 1.0}) == pytest.approx(1.0)
    assert compute_hhi_concentration({}) == 0.0


def test_compute_qoq_turnover_and_position_buckets():
    q3 = Filing13F(
        cik="0001",
        manager_name="FUND A",
        period_of_report="2025-09-30",
        filing_date="2025-11-14",
        holdings=[
            HoldingRecord(cusip="037833100", ticker="AAPL", issuer_name="APPLE", title_of_class="COM", value_usd=50.0, shares=100),
            HoldingRecord(cusip="594918104", ticker="MSFT", issuer_name="MSFT", title_of_class="COM", value_usd=50.0, shares=100),
        ],
    )
    q4 = Filing13F(
        cik="0001",
        manager_name="FUND A",
        period_of_report="2025-12-31",
        filing_date="2026-02-14",
        holdings=[
            HoldingRecord(cusip="037833100", ticker="AAPL", issuer_name="APPLE", title_of_class="COM", value_usd=30.0, shares=60),
            HoldingRecord(cusip="67066G104", ticker="NVDA", issuer_name="NVIDIA", title_of_class="COM", value_usd=70.0, shares=500),
        ],
    )
    report = compute_qoq_turnover(q3, q4)
    # AAPL went 0.50 -> 0.30 (-0.20), MSFT went 0.50 -> 0.00 (-0.50), NVDA went 0.00 -> 0.70 (+0.70)
    # One-way turnover = 0.5 * (0.20 + 0.50 + 0.70) = 0.70
    assert report.turnover_ratio == pytest.approx(0.70)
    assert report.initiated_tickers == ["NVDA"]
    assert report.exited_tickers == ["MSFT"]
    assert report.trimmed_tickers == ["AAPL"]


def test_parse_form4_xml_and_insider_cluster_scoring():
    txs = parse_form4_xml(SAMPLE_FORM4_XML)
    assert len(txs) == 1
    assert txs[0].issuer_ticker == "NVDA"
    assert txs[0].is_officer is True
    assert txs[0].notional_usd == pytest.approx(1_305_000.0)

    # Add a second officer open-market purchase to trigger cluster bonus
    tx2 = InsiderTransaction(
        cik="0001045810",
        issuer_ticker="NVDA",
        insider_name="SMITH JOHN",
        is_officer=True,
        is_director=True,
        is_ten_pct_owner=False,
        transaction_code="P",
        shares=5000,
        price_per_share=131.0,
        acquired_disposed="A",
        transaction_date="2026-02-21",
    )
    # And an open-market sale on TSLA
    tx3 = InsiderTransaction(
        cik="0001318605",
        issuer_ticker="TSLA",
        insider_name="EXEC ONE",
        is_officer=True,
        is_director=False,
        is_ten_pct_owner=False,
        transaction_code="S",
        shares=2000,
        price_per_share=250.0,
        acquired_disposed="D",
        transaction_date="2026-02-21",
    )
    scores = compute_insider_cluster_score([txs[0], tx2, tx3])
    assert scores["NVDA"] > 35.0
    assert scores["TSLA"] < 0.0


def _sample_multi_fund_filings() -> list[Filing13F]:
    f1 = Filing13F(
        cik="1",
        manager_name="TIGER GLOBAL",
        period_of_report="2025-12-31",
        filing_date="2026-02-14",
        holdings=[
            HoldingRecord("67066G104", "NVDA", "NVIDIA", "COM", 600.0, 4000),
            HoldingRecord("594918104", "MSFT", "MICROSOFT", "COM", 300.0, 750),
            HoldingRecord("037833100", "AAPL", "APPLE", "COM", 100.0, 500),
        ],
    )
    f2 = Filing13F(
        cik="2",
        manager_name="COATUE",
        period_of_report="2025-12-31",
        filing_date="2026-02-14",
        holdings=[
            HoldingRecord("67066G104", "NVDA", "NVIDIA", "COM", 500.0, 3500),
            HoldingRecord("594918104", "MSFT", "MICROSOFT", "COM", 300.0, 750),
            HoldingRecord("30303M102", "META", "META", "COM", 200.0, 400),
        ],
    )
    f3 = Filing13F(
        cik="3",
        manager_name="LONE PINE",
        period_of_report="2025-12-31",
        filing_date="2026-02-14",
        holdings=[
            HoldingRecord("67066G104", "NVDA", "NVIDIA", "COM", 700.0, 5000),
            HoldingRecord("023135106", "AMZN", "AMAZON", "COM", 300.0, 1500),
        ],
    )
    return [f1, f2, f3]


def test_compute_crowding_scores_and_z_scores():
    filings = _sample_multi_fund_filings()
    floats = {"NVDA": 25_000, "MSFT": 10_000, "AAPL": 20_000, "META": 10_000, "AMZN": 15_000}
    metrics = compute_crowding_scores(filings, float_shares_by_ticker=floats)

    # NVDA is held by 100% of funds with highest average weight -> highest positive crowding z-score
    assert metrics["NVDA"].holder_count == 3
    assert metrics["NVDA"].ownership_breadth == pytest.approx(1.0)
    assert metrics["NVDA"].pct_of_float == pytest.approx(12_500 / 25_000)
    assert metrics["NVDA"].crowding_z_score > 1.0

    # Mean of cross-sectional z-scores across the 5 tickers must be 0.0
    z_sum = sum(m.crowding_z_score for m in metrics.values())
    assert z_sum == pytest.approx(0.0, abs=1e-5)


def test_construct_clone_portfolio_modes():
    filings = _sample_multi_fund_filings()
    eq = construct_clone_portfolio(filings, mode="equal", top_n=2)
    assert set(eq.keys()) == {"NVDA", "MSFT"}
    assert eq["NVDA"] == pytest.approx(0.5)
    assert eq["MSFT"] == pytest.approx(0.5)

    conv = construct_clone_portfolio(filings, mode="conviction", top_n=3)
    assert sum(conv.values()) == pytest.approx(1.0)
    assert conv["NVDA"] > conv["MSFT"]

    anti = construct_clone_portfolio(filings, mode="anti_crowded_conviction", top_n=3)
    assert sum(anti.values()) == pytest.approx(1.0)
    # Anti-crowded mode reduces NVDA's weight relative to pure conviction weighting
    assert anti["NVDA"] < conv["NVDA"]


def test_backtest_13f_strategy_alpha_and_sharpe():
    filings = _sample_multi_fund_filings()
    quarterly_filings = {
        "2025-Q1": filings,
        "2025-Q2": filings,
        "2025-Q3": filings,
        "2025-Q4": filings,
    }
    post_filing_returns = {
        "2025-Q1": {"NVDA": 0.12, "MSFT": 0.06, "AMZN": 0.05, "META": 0.04, "AAPL": 0.03},
        "2025-Q2": {"NVDA": 0.08, "MSFT": 0.05, "AMZN": 0.04, "META": 0.03, "AAPL": 0.02},
        "2025-Q3": {"NVDA": -0.02, "MSFT": 0.01, "AMZN": 0.02, "META": 0.01, "AAPL": 0.00},
        "2025-Q4": {"NVDA": 0.15, "MSFT": 0.07, "AMZN": 0.06, "META": 0.05, "AAPL": 0.04},
    }
    benchmark_returns = {
        "2025-Q1": 0.04,
        "2025-Q2": 0.03,
        "2025-Q3": -0.01,
        "2025-Q4": 0.05,
    }

    res = backtest_13f_strategy(
        quarterly_filings=quarterly_filings,
        post_filing_period_returns=post_filing_returns,
        benchmark_returns=benchmark_returns,
        mode="conviction",
        top_n=3,
        risk_free_rate_annual=0.04,
    )
    assert res.cumulative_return > 0.20
    assert res.sharpe_ratio > 1.5
    assert res.jensen_alpha_annualized > 0.05
    assert res.information_ratio > 1.0
    assert 0.0 <= res.max_drawdown < 0.05


def test_invalid_clone_mode_and_empty_backtest_rejected():
    filings = _sample_multi_fund_filings()
    with pytest.raises(ValueError, match="Unsupported clone portfolio mode"):
        construct_clone_portfolio(filings, mode="invalid_mode")
    with pytest.raises(ValueError, match="quarterly_filings cannot be empty"):
        backtest_13f_strategy({}, {}, {})


def test_parse_13f_xml_value_in_thousands_scaling():
    # Legacy pre-2023 13F filings report values in $1,000s
    filing = parse_13f_xml(SAMPLE_13F_XML_NAMESPACED, value_in_thousands=True)
    assert filing.total_value_usd == pytest.approx(105_000_000_000.0)


def test_parse_13f_xml_missing_infotable_raises():
    with pytest.raises(ValueError, match="No <infoTable> elements found"):
        parse_13f_xml("<informationTable></informationTable>")


def test_compute_qoq_turnover_increased_positions():
    q1 = Filing13F(
        cik="0009",
        manager_name="FUND B",
        period_of_report="2025-03-31",
        filing_date="2025-05-15",
        holdings=[
            HoldingRecord("037833100", "AAPL", "APPLE", "COM", 40.0, 200),
            HoldingRecord("594918104", "MSFT", "MICROSOFT", "COM", 60.0, 150),
        ],
    )
    q2 = Filing13F(
        cik="0009",
        manager_name="FUND B",
        period_of_report="2025-06-30",
        filing_date="2025-08-14",
        holdings=[
            HoldingRecord("037833100", "AAPL", "APPLE", "COM", 70.0, 350),
            HoldingRecord("594918104", "MSFT", "MICROSOFT", "COM", 30.0, 75),
        ],
    )
    rep = compute_qoq_turnover(q1, q2)
    assert rep.increased_tickers == ["AAPL"]
    assert rep.trimmed_tickers == ["MSFT"]
    assert rep.turnover_ratio == pytest.approx(0.30)
