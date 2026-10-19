# sec-edgar-13f-alpha

Quantitative **SEC EDGAR Form 13F-HR** institutional holdings & **Form 4** insider transaction parser, **ANSI X9.6 CUSIP Mod-10** check-digit validator, cross-sectional **institutional crowding z-score** engine, and **point-in-time clone portfolio alpha backtester**.

---

## Key Capabilities

1. **ANSI X9.6 CUSIP Mod-10 Check-Digit Validation & Ticker Resolution**
   - Implements the "Double-Add-Double" modulus-10 check-digit algorithm across numeric, alphabetic (`A=10..Z=35`), and PPN special characters (`*`, `@`, `#`).
   - Maps 9-character CUSIPs to canonical equity tickers with customizable CUSIP registration and issuer-name normalization fallback.

2. **SEC EDGAR Form 13F-HR XML/JSON & Form 4 Ownership Parser**
   - Parses namespaced and un-namespaced SEC EDGAR `informationTable` XML filings.
   - Consolidates split sub-manager `<infoTable>` rows by `(cusip, putCall, sshPrnamtType)` and separates common stock (`SH`) from listed `PUT`/`CALL` option positions.
   - Parses Form 4 `ownershipDocument` XML into structured open-market insider buys (`P`) and sales (`S`), computing role-weighted and breadth-adjusted insider cluster conviction scores.

3. **Portfolio Concentration, QoQ Turnover & Institutional Crowding Z-Scores**
   - Computes Herfindahl-Hirschman Index ($\text{HHI} = \sum w_i^2$) concentration and one-way quarter-over-quarter portfolio turnover ($\frac{1}{2}\sum |w_{i,t} - w_{i,t-1}|$), partitioning transitions into initiated, exited, increased, and trimmed positions.
   - Computes cross-sectional institutional crowding z-scores combining fund ownership breadth, average manager conviction weight, and percentage of float held.

4. **Temporal 13F Clone Portfolio Construction & Alpha Attribution**
   - Supports Equal-Weighted (`equal`), Conviction-Weighted (`conviction`), and Anti-Crowded Conviction (`anti_crowded_conviction`) clone portfolios.
   - Evaluates point-in-time returns post 45-day statutory SEC filing lag, reporting annualized return, volatility, Sharpe ratio, OLS beta, annualized Jensen's $\alpha$, Information Ratio, and maximum drawdown.

---

## Quick Start

```python
from edgar_13f import (
    compute_crowding_scores,
    construct_clone_portfolio,
    parse_13f_xml,
    validate_cusip,
)

assert validate_cusip("67066G104")  # NVDA

filing = parse_13f_xml(xml_string, cik="0001067983", manager_name="QUANT VALUE LP")
crowding = compute_crowding_scores([filing])
weights = construct_clone_portfolio([filing], mode="anti_crowded_conviction", top_n=10)
```

---

## Running Tests

```bash
pytest -v
```

---

## Author

**Abdullah Azmi**  
- Website: [https://abdullahazmi.com](https://abdullahazmi.com)  
- GitHub: [@aazminu](https://github.com/aazminu)  
- Email: `m4u5t4@u.northwestern.edu`
