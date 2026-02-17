"""
Zakat Calculator Data Pipeline
==============================
Fetches balance sheets from Yahoo Finance for all fund + ISA holdings,
computes zakaatable percentages using actual balance sheet data,
and generates self-contained HTML calculator pages.

Methodology:
  Zakaatable Assets = Cash & STI + Receivables + Inventory + Other Current Assets
  Deductible Liabilities = Payables + Current Debt (excl. leases) + Accrued Expenses + Taxes Payable

  NOT deducted: deferred revenue, lease liabilities, long-term debt, deferred taxes,
                pension obligations, provisions, OtherCurrentLiabilities (too opaque)

  Strict Net Zakaatable = max(Current Assets - Deductible Liabilities, 0)
  Broad Net Zakaatable = max(All Liquid Assets - Deductible Liabilities, 0)
  Assets-Only Zakaatable = All Liquid Assets (no liability deductions)
  Zakaatable % = min(Method Amount / Market Cap * 100, 100)

  Fund Zakaatable % = weighted sum of company zakaatable % + cash allocation
  Zakat = Value * Fund Zakaatable % * 2.5%
"""

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yfinance as yf
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ──────────────────────────────────────────────
# 1. FX RATES
# ──────────────────────────────────────────────

def fetch_fx_rates():
    """Fetch live FX rates from ExchangeRate-API (free, no key needed)."""
    print("\n[1/4] Fetching live FX rates...")
    url = "https://open.er-api.com/v6/latest/USD"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("result") != "success":
        raise RuntimeError(f"FX API returned error: {data}")

    rates = data["rates"]  # rates relative to 1 USD

    # Add GBp / GBX (British pence: 100p = 1 GBP)
    gbp_rate = rates.get("GBP", 1)
    rates["GBp"] = gbp_rate * 100  # 1 USD = X GBp
    rates["GBX"] = gbp_rate * 100

    output = {
        "base": "USD",
        "rates": rates,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }

    out_path = DATA_DIR / "fx_rates.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(rates)} currency rates to {out_path}")
    return rates


# ──────────────────────────────────────────────
# 2. BALANCE SHEET FETCHER
# ──────────────────────────────────────────────

def safe_get(series, *field_names, default=0.0):
    """Try each field name in order, return first non-null numeric value.
    Handles both PascalCase and space-separated yfinance field names."""
    for name in field_names:
        # Try exact match first
        if name in series.index:
            val = series[name]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
        # Try space-separated version: "CashAndCashEquivalents" -> "Cash And Cash Equivalents"
        spaced = ""
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                spaced += " "
            spaced += ch
        if spaced != name and spaced in series.index:
            val = series[spaced]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return float(val)
    return default


def fetch_company(ticker_symbol):
    """Fetch balance sheet and market data for a single ticker."""
    ticker = yf.Ticker(ticker_symbol)

    # Get info first for market cap and currencies
    try:
        info = ticker.info
    except Exception:
        info = {}

    market_cap = info.get("marketCap") or 0
    trading_currency = info.get("currency", "USD")
    financial_currency = info.get("financialCurrency", trading_currency)
    current_price = info.get("currentPrice") or info.get("previousClose") or 0
    shares_outstanding = info.get("sharesOutstanding") or 0
    long_name = info.get("longName") or info.get("shortName") or ticker_symbol

    # If market cap is missing, try computing from shares * price
    if not market_cap and shares_outstanding and current_price:
        market_cap = shares_outstanding * current_price

    # Prefer quarterly balance sheet (more recent), fall back to annual
    bs = None
    bs_type = "quarterly"
    try:
        qbs = ticker.quarterly_balance_sheet
        if qbs is not None and not qbs.empty:
            bs = qbs
    except Exception:
        pass

    if bs is None:
        bs_type = "annual"
        try:
            abs_ = ticker.balance_sheet
            if abs_ is not None and not abs_.empty:
                bs = abs_
        except Exception:
            pass

    if bs is None or bs.empty:
        return {
            "ticker": ticker_symbol,
            "long_name": long_name,
            "market_cap": market_cap,
            "trading_currency": trading_currency,
            "financial_currency": financial_currency,
            "current_price": current_price,
            "error": "No balance sheet data available",
            "zakaatable_pct": 25.0,  # fallback estimate
            "fallback": True,
        }

    latest = bs.iloc[:, 0]
    bs_date = str(bs.columns[0].date()) if hasattr(bs.columns[0], "date") else str(bs.columns[0])

    # ── ZAKAATABLE ASSETS ──
    cash_sti = safe_get(latest, "CashCashEquivalentsAndShortTermInvestments")
    if cash_sti == 0:
        cash_sti = (
            safe_get(latest, "CashAndCashEquivalents", "CashEquivalents")
            + safe_get(latest, "OtherShortTermInvestments", "AvailableForSaleSecurities")
        )

    receivables = safe_get(latest, "Receivables", "AccountsReceivable", "NetReceivables")
    inventory = safe_get(latest, "Inventory", "Inventories")
    other_current = safe_get(latest, "OtherCurrentAssets", "PrepaidAssets")

    gross_zakaatable = cash_sti + receivables + inventory + other_current

    # ── NON-CURRENT LIQUID ASSETS (for broad/IFG method) ──
    # These are financial investments held long-term but still liquid/sellable
    nc_investments = safe_get(latest, "InvestmentsAndAdvances", "InvestmentinFinancialAssets")
    nc_deferred_tax_asset = safe_get(latest, "NonCurrentDeferredTaxesAssets", "NonCurrentDeferredAssets")
    nc_receivables = safe_get(latest, "NonCurrentAccountsReceivable")
    nc_liquid_total = nc_investments + nc_deferred_tax_asset + nc_receivables

    gross_zakaatable_broad = gross_zakaatable + nc_liquid_total

    # ── DEDUCTIBLE LIABILITIES ──
    # Use AccountsPayable specifically (not the parent "Payables" which may include taxes/accrued)
    payables = safe_get(latest, "AccountsPayable")

    # Current debt: try to exclude lease obligations
    current_debt = safe_get(latest, "CurrentDebt")
    if current_debt == 0:
        combined = safe_get(latest, "CurrentDebtAndCapitalLeaseObligation")
        lease = safe_get(latest, "CurrentCapitalLeaseObligation")
        current_debt = max(combined - lease, 0)

    accrued = safe_get(latest, "CurrentAccruedExpenses")
    taxes = safe_get(latest, "TotalTaxPayable", "IncomeTaxPayable", "TaxesPayable")

    total_deductions = payables + current_debt + accrued + taxes
    net_zakaatable_strict = max(gross_zakaatable - total_deductions, 0)
    net_zakaatable_broad = max(gross_zakaatable_broad - total_deductions, 0)
    net_zakaatable_assets_only = gross_zakaatable_broad

    # If all balance sheet values are 0 (all NaN), treat as missing data
    all_zero = (cash_sti == 0 and receivables == 0 and inventory == 0
                and other_current == 0 and payables == 0 and current_debt == 0)
    if all_zero:
        return {
            "ticker": ticker_symbol,
            "long_name": long_name,
            "market_cap": market_cap,
            "trading_currency": trading_currency,
            "financial_currency": financial_currency,
            "current_price": current_price,
            "error": "Balance sheet data all empty/NaN",
            "zakaatable_pct": 25.0,
            "fallback": True,
        }

    return {
        "ticker": ticker_symbol,
        "long_name": long_name,
        "market_cap": market_cap,
        "trading_currency": trading_currency,
        "financial_currency": financial_currency,
        "current_price": current_price,
        "bs_date": bs_date,
        "bs_type": bs_type,
        "assets": {
            "cash_and_sti": cash_sti,
            "receivables": receivables,
            "inventory": inventory,
            "other_current": other_current,
            "total": gross_zakaatable,
        },
        "assets_broad": {
            "nc_investments": nc_investments,
            "nc_deferred_tax_asset": nc_deferred_tax_asset,
            "nc_receivables": nc_receivables,
            "nc_liquid_total": nc_liquid_total,
            "total": gross_zakaatable_broad,
        },
        "liabilities": {
            "payables": payables,
            "current_debt": current_debt,
            "accrued_expenses": accrued,
            "taxes_payable": taxes,
            "total": total_deductions,
        },
        "net_zakaatable": net_zakaatable_strict,
        "net_zakaatable_broad": net_zakaatable_broad,
        "net_zakaatable_assets_only": net_zakaatable_assets_only,
        "error": None,
        "fallback": False,
    }


def fetch_all_balance_sheets(tickers):
    """Fetch balance sheets for all tickers with rate limiting."""
    print(f"\n[2/4] Fetching balance sheets for {len(tickers)} unique tickers...")
    results = {}
    errors = []

    for i, t in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {t}...", end=" ", flush=True)

        try:
            data = fetch_company(t)
            results[t] = data
            if data.get("error"):
                print(f"WARNING: {data['error']}")
                errors.append((t, data["error"]))
            else:
                print(f"OK (zakaatable assets: {data['net_zakaatable']:,.0f})")
        except Exception as e:
            print(f"ERROR: {e}")
            results[t] = {
                "ticker": t,
                "error": str(e),
                "zakaatable_pct": 25.0,
                "fallback": True,
            }
            errors.append((t, str(e)))

        if i < len(tickers) - 1:
            time.sleep(1.5)

    out_path = DATA_DIR / "balance_sheets.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved balance sheet data to {out_path}")

    if errors:
        print(f"\n  {len(errors)} tickers had issues:")
        for t, e in errors:
            print(f"    {t}: {e}")

    return results


# ──────────────────────────────────────────────
# 3. ZAKAT COMPUTATION
# ──────────────────────────────────────────────

def _convert_and_calc_pct(net_val, market_cap, fin_curr, trade_curr, fx_rates):
    """Convert net_zakaatable to market cap currency and compute %."""
    mcap_curr = trade_curr
    if mcap_curr in ("GBp", "GBX"):
        mcap_curr = "GBP"

    if fin_curr != mcap_curr:
        fin_rate = fx_rates.get(fin_curr, 1)
        mcap_rate = fx_rates.get(mcap_curr, 1)
        if fin_rate > 0 and mcap_rate > 0:
            net_val = net_val / fin_rate * mcap_rate

    pct = (net_val / market_cap) * 100
    return round(max(0, min(pct, 100)), 4)


def compute_zakaatable_pcts(company_data, fx_rates):
    """Compute strict, broad, and assets-only zakaatable % for a company.
    Returns (strict_pct, broad_pct, assets_only_pct)."""
    if company_data.get("fallback"):
        fb = company_data.get("zakaatable_pct", 25.0)
        return fb, fb, fb

    market_cap = company_data.get("market_cap", 0)
    if not market_cap or market_cap <= 0:
        return 25.0, 25.0, 25.0

    fin_curr = company_data.get("financial_currency", "USD")
    trade_curr = company_data.get("trading_currency", "USD")

    net_strict = company_data.get("net_zakaatable", 0)
    net_broad = company_data.get("net_zakaatable_broad", net_strict)
    assets_only = company_data.get(
        "net_zakaatable_assets_only",
        company_data.get("assets_broad", {}).get(
            "total",
            company_data.get("assets", {}).get("total", net_broad),
        ),
    )

    pct_strict = _convert_and_calc_pct(net_strict, market_cap, fin_curr, trade_curr, fx_rates)
    pct_broad = _convert_and_calc_pct(net_broad, market_cap, fin_curr, trade_curr, fx_rates)
    pct_assets_only = _convert_and_calc_pct(assets_only, market_cap, fin_curr, trade_curr, fx_rates)

    return pct_strict, pct_broad, pct_assets_only


def compute_pension_data(fund_holdings, balance_sheets, fx_rates):
    """Compute zakaatable data for all pension fund holdings."""
    print("\n[3/4] Computing zakaatable percentages...")

    results = []
    for h in fund_holdings["holdings"]:
        ticker = h["ticker"]
        bs_data = balance_sheets.get(ticker, {})

        pct_strict, pct_broad, pct_assets_only = compute_zakaatable_pcts(bs_data, fx_rates)

        entry = {
            "name": h["name"],
            "ticker": ticker,
            "country": h["country"],
            "weight": h["weight"],
            "zakaatable_pct": pct_strict,
            "zakaatable_pct_broad": pct_broad,
            "zakaatable_pct_assets_only": pct_assets_only,
            "fallback": bs_data.get("fallback", True),
            "error": bs_data.get("error"),
            "bs_date": bs_data.get("bs_date", "N/A"),
            "market_cap": bs_data.get("market_cap", 0),
            "trading_currency": bs_data.get("trading_currency", "USD"),
            "financial_currency": bs_data.get("financial_currency", "USD"),
            "assets": bs_data.get("assets", {}),
            "assets_broad": bs_data.get("assets_broad", {}),
            "liabilities": bs_data.get("liabilities", {}),
            "net_zakaatable": bs_data.get("net_zakaatable", 0),
            "net_zakaatable_broad": bs_data.get("net_zakaatable_broad", 0),
            "net_zakaatable_assets_only": bs_data.get(
                "net_zakaatable_assets_only",
                bs_data.get("assets_broad", {}).get(
                    "total",
                    bs_data.get("assets", {}).get("total", 0),
                ),
            ),
        }
        results.append(entry)

        status = "FALLBACK" if entry["fallback"] else "OK"
        print(
            f"  {h['name']:<40} weight={h['weight']:5.2f}%  "
            f"strict={pct_strict:6.2f}%  broad={pct_broad:6.2f}%  assets_only={pct_assets_only:6.2f}%  [{status}]"
        )

    # Compute fund-level zakaatable %
    cash_contrib = fund_holdings["fund_cash_pct"]  # cash is 100% zakaatable
    fund_pct_strict = sum(r["weight"] * r["zakaatable_pct"] / 100 for r in results) + cash_contrib
    fund_pct_broad = sum(r["weight"] * r["zakaatable_pct_broad"] / 100 for r in results) + cash_contrib
    fund_pct_assets_only = sum(r["weight"] * r["zakaatable_pct_assets_only"] / 100 for r in results) + cash_contrib

    pension_data = {
        "fund_name": fund_holdings["fund_name"],
        "benchmark": fund_holdings["benchmark"],
        "report_date": fund_holdings["report_date"],
        "fund_cash_pct": fund_holdings["fund_cash_pct"],
        "fund_zakaatable_pct": round(fund_pct_strict, 4),
        "fund_zakaatable_pct_broad": round(fund_pct_broad, 4),
        "fund_zakaatable_pct_assets_only": round(fund_pct_assets_only, 4),
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "holdings": results,
    }

    out_path = DATA_DIR / "pension_zakat.json"
    with open(out_path, "w") as f:
        json.dump(pension_data, f, indent=2)

    print(
        "\n  Fund zakaatable %:  "
        f"strict={fund_pct_strict:.4f}%  broad={fund_pct_broad:.4f}%  assets_only={fund_pct_assets_only:.4f}%"
    )
    print(f"  (vs common 25% estimate)")
    print(f"  Saved to {out_path}")

    return pension_data


def compute_isa_data(isa_holdings, balance_sheets, fx_rates):
    """Compute zakaatable data for ISA holdings."""
    results = []
    for h in isa_holdings["holdings"]:
        ticker = h["ticker"]
        bs_data = balance_sheets.get(ticker, {})

        pct_strict, pct_broad, pct_assets_only = compute_zakaatable_pcts(bs_data, fx_rates)

        # Get current price and convert to GBP
        current_price = bs_data.get("current_price", 0)
        trade_curr = bs_data.get("trading_currency", "USD")

        # Price in GBP
        gbp_rate = fx_rates.get("GBP", 1)
        trade_rate = fx_rates.get(trade_curr, 1)
        price_gbp = 0
        if trade_rate > 0:
            price_gbp = current_price / trade_rate * gbp_rate

        # Handle GBp: price is in pence, convert to pounds
        if trade_curr in ("GBp", "GBX"):
            price_gbp = current_price / 100

        entry = {
            "name": h["name"],
            "ticker": ticker,
            "zakaatable_pct": pct_strict,
            "zakaatable_pct_broad": pct_broad,
            "zakaatable_pct_assets_only": pct_assets_only,
            "current_price": current_price,
            "price_gbp": round(price_gbp, 4),
            "trading_currency": trade_curr,
            "fallback": bs_data.get("fallback", True),
            "error": bs_data.get("error"),
            "bs_date": bs_data.get("bs_date", "N/A"),
            "market_cap": bs_data.get("market_cap", 0),
            "financial_currency": bs_data.get("financial_currency", "USD"),
            "assets": bs_data.get("assets", {}),
            "assets_broad": bs_data.get("assets_broad", {}),
            "liabilities": bs_data.get("liabilities", {}),
            "net_zakaatable": bs_data.get("net_zakaatable", 0),
            "net_zakaatable_broad": bs_data.get("net_zakaatable_broad", 0),
            "net_zakaatable_assets_only": bs_data.get(
                "net_zakaatable_assets_only",
                bs_data.get("assets_broad", {}).get(
                    "total",
                    bs_data.get("assets", {}).get("total", 0),
                ),
            ),
        }
        results.append(entry)

    isa_data = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "holdings": results,
    }

    out_path = DATA_DIR / "isa_zakat.json"
    with open(out_path, "w") as f:
        json.dump(isa_data, f, indent=2)

    print(f"\n  ISA data saved to {out_path}")
    return isa_data


# ──────────────────────────────────────────────
# 4. HTML GENERATION
# ──────────────────────────────────────────────

def build_pension_html(pension_data):
    """Generate pension.html with embedded data."""
    json_str = json.dumps(pension_data, indent=2)

    template_path = BASE_DIR / "pension_template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace("/* __PENSION_DATA__ */", json_str)

    out_path = BASE_DIR / "pension.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Built {out_path}")


def build_isa_html(isa_data):
    """Generate isa.html with embedded data."""
    json_str = json.dumps(isa_data, indent=2)

    template_path = BASE_DIR / "isa_template.html"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace("/* __ISA_DATA__ */", json_str)

    out_path = BASE_DIR / "isa.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Built {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZAKAT CALCULATOR - DATA PIPELINE")
    print("=" * 60)

    # Load holdings
    with open(DATA_DIR / "fund_holdings.json") as f:
        fund_holdings = json.load(f)
    with open(DATA_DIR / "isa_holdings.json") as f:
        isa_holdings = json.load(f)

    # Collect unique tickers
    fund_tickers = [h["ticker"] for h in fund_holdings["holdings"]]
    isa_tickers = [h["ticker"] for h in isa_holdings["holdings"]]
    all_tickers = list(dict.fromkeys(fund_tickers + isa_tickers))  # dedupe, preserve order

    print(f"\n  Fund holdings: {len(fund_holdings['holdings'])}")
    print(f"  ISA holdings:  {len(isa_holdings['holdings'])}")
    print(f"  Unique tickers: {len(all_tickers)}")

    # Step 1: FX rates
    fx_rates = fetch_fx_rates()

    # Step 2: Balance sheets
    balance_sheets = fetch_all_balance_sheets(all_tickers)

    # Step 3: Compute zakat
    pension_data = compute_pension_data(fund_holdings, balance_sheets, fx_rates)
    isa_data = compute_isa_data(isa_holdings, balance_sheets, fx_rates)

    # Step 4: Build HTML
    print("\n[4/4] Building HTML files...")
    build_pension_html(pension_data)
    build_isa_html(isa_data)

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print(f"  Fund zakaatable: {pension_data['fund_zakaatable_pct']:.2f}%")
    print(f"  Open pension.html and isa.html in your browser")
    print("=" * 60)


if __name__ == "__main__":
    main()
