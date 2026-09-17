import os
import json
import io
import datetime as dt
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from data_quality import audit_ohlcv

# Optional local secrets (not committed). Create secrets_local.py with
# ALPHAVANTAGE_API_KEY = "..." to use.
try:
    from secrets_local import ALPHAVANTAGE_API_KEY as _KEY_LOCAL  # type: ignore
except Exception:  # pragma: no cover
    _KEY_LOCAL = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# Optional HTTP client for Alpha Vantage
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


def _validated_ohlcv(df: pd.DataFrame, *, source: str, auto_adjust: bool) -> pd.DataFrame:
    """Run one provider result through the shared market-data audit."""
    # Invalid rows are removed and counted rather than silently filled. This
    # keeps a single bad provider row from discarding an otherwise valid range.
    clean, report = audit_ohlcv(df, strict=False)
    report = dict(report)
    report.update({"source": source, "auto_adjust": bool(auto_adjust)})
    clean.attrs["data_quality"] = report
    return clean


def get_current_price(
    ticker_yt: str = "2330.TW",
    ticker_sj: str = "2330",
) -> Tuple[Optional[float], str]:
    """Get current/last price for TSMC.

    Tries Yahoo Finance first; falls back to Alpha Vantage if configured.
    Returns: (price or None, source string)
    """
    # Yahoo Finance
    if yf is not None:
        try:
            info = yf.Ticker(ticker_yt).info
            price = info.get("currentPrice", None)
            if price is not None:
                return float(price), "Yahoo Finance"
        except Exception:
            pass

    # Alpha Vantage fallback (no login in code, requires API key)
    price = _fetch_alpha_vantage_quote(ticker_yt)
    if price is not None:
        return price, "Alpha Vantage"

    # Shioaji fallback removed

    return None, "None"


def get_ohlcv(
    ticker_yt: str = "2330.TW",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance.

    Returns a DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    If Yahoo is unavailable, falls back to Alpha Vantage (requires API key).
    """
    # If explicitly using Alpha Vantage only
    if provider and provider.lower() == "alpha":
        df_av = _fetch_alpha_vantage_ohlcv(ticker_yt, start=start, end=end, auto_adjust=auto_adjust)
        if df_av is not None and not df_av.empty:
            return _validated_ohlcv(df_av, source="alpha_vantage", auto_adjust=auto_adjust)
        raise RuntimeError("Alpha Vantage provider selected but no data returned.")

    def _fetch_yf() -> Optional[pd.DataFrame]:
        if yf is None:
            return None
        try:
            # yfinance treats 'end' as exclusive. Make UI/API 'end' inclusive by adding one day.
            end_param = end
            try:
                if end is not None:
                    end_dt = pd.to_datetime(end)
                    end_param = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            except Exception:
                end_param = end
            df = yf.download(
                ticker_yt,
                start=start,
                end=end_param,
                auto_adjust=auto_adjust,
                group_by="column",
                progress=False,
            )
            # Flatten possible MultiIndex columns (yfinance may return them)
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance has used both (field, ticker) and (ticker, field).
                # Select the level that actually contains OHLCV labels instead
                # of assuming a fixed level order.
                expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
                scores = []
                for level in range(df.columns.nlevels):
                    labels = {str(value) for value in df.columns.get_level_values(level)}
                    scores.append((len(labels & expected), level))
                best_score, best_level = max(scores)
                if best_score:
                    df.columns = df.columns.get_level_values(best_level)
                else:
                    df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip("_") for tup in df.columns]
            # Normalize column title case for consistency
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj close": "Adj Close",
                "volume": "Volume",
            })
            # Ensure a Close column exists (fallback to Adj Close if necessary)
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            return df
        except Exception:
            return None

    # Primary: Yahoo Finance
    df_yf = _fetch_yf() if (provider is None or provider.lower() == "yahoo") else None

    # TWSE fallback for .TW tickers when yfinance has gaps
    def _fetch_twse(tw_ticker: str) -> Optional[pd.DataFrame]:
        try:
            sym = tw_ticker.replace(".TW", "")
            # TWSE provides daily quotes CSV per date; here we attempt a simple contiguous fetch via period=all
            # Note: this endpoint may change; it's a best-effort fallback.
            url = f"https://www.twse.com.tw/rwd/en/afterTrading/STOCK_DAY?response=json&date=&stockNo={sym}"
            if requests is None:
                return None
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data or "data" not in data:
                return None
            cols = data.get("fields", [])
            rows = data.get("data", [])
            df = pd.DataFrame(rows, columns=cols)
            # Normalize columns
            colmap = {
                "Date": "Date",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
            # TWSE uses numeric strings with commas
            for c in ["Open", "High", "Low", "Close"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",", ""), errors="coerce")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            return None

    # Do not mix Yahoo's adjusted bars with TWSE's unadjusted bars. Validate one
    # complete source and only fall back when the preferred source is absent.
    if df_yf is not None and not df_yf.empty:
        return _validated_ohlcv(df_yf, source="yahoo_finance", auto_adjust=auto_adjust)

    df_twse = _fetch_twse(ticker_yt) if ticker_yt.upper().endswith(".TW") else None
    if df_twse is not None and not df_twse.empty:
        return _validated_ohlcv(df_twse, source="twse", auto_adjust=False)

    # Secondary: Alpha Vantage (requires API key)
    df_av = _fetch_alpha_vantage_ohlcv(ticker_yt, start=start, end=end, auto_adjust=auto_adjust)
    if df_av is not None and not df_av.empty:
        return _validated_ohlcv(df_av, source="alpha_vantage", auto_adjust=auto_adjust)

    raise RuntimeError("Unable to fetch OHLCV from Yahoo or Alpha Vantage.")


def _norm_name(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .strip()
    )


def _extract_row(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    """Extract a row by any of the candidate names (robust to format variations).

    Returns a Series indexed by the df's columns (usually period end dates), or None.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        idx_map = { _norm_name(i): i for i in df.index }
        for cand in candidates:
            key = _norm_name(cand)
            if key in idx_map:
                s = df.loc[idx_map[key]]
                # Ensure numeric
                s = pd.to_numeric(s, errors="coerce")
                s.name = candidates[0]
                return s
    except Exception:
        return None
    return None


def _safe_df(obj) -> Optional[pd.DataFrame]:
    try:
        if obj is None:
            return None
        df = obj
        if isinstance(df, pd.DataFrame) and not df.empty:
            # yfinance quarterly_* uses columns as period ends
            # Ensure columns are datetime
            try:
                df.columns = pd.to_datetime(df.columns)
            except Exception:
                pass
            return df
    except Exception:
        return None
    return None


def get_fundamentals_timeseries(
    ticker: str,
    align_index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Fetch key Yahoo Finance fundamentals as a quarterly time series.

    Returns a DataFrame indexed by the reported period end (ascending) with
    columns prefixed by 'f_' for fundamentals. If align_index is provided,
    reindexes to that daily index via forward-fill (no backfill to avoid leakage).
    """
    if yf is None:
        return pd.DataFrame(index=(align_index if align_index is not None else []))

    try:
        t = yf.Ticker(ticker)
        fin_q = _safe_df(getattr(t, "quarterly_financials", None))
        bs_q = _safe_df(getattr(t, "quarterly_balance_sheet", None))
        cf_q = _safe_df(getattr(t, "quarterly_cashflow", None))
    except Exception:
        fin_q = bs_q = cf_q = None

    # Core series
    s_rev = _extract_row(fin_q, ["total revenue", "totalrevenue"])  # revenue
    s_gross = _extract_row(fin_q, ["gross profit", "grossprofit"])  # gross profit
    s_opinc = _extract_row(fin_q, ["operating income", "operatingincome"])  # operating income
    s_netinc = _extract_row(fin_q, ["net income", "netincome"])  # net income
    s_ebitda = _extract_row(fin_q, ["ebitda"])  # EBITDA

    s_tot_assets = _extract_row(bs_q, ["total assets", "totalassets"])  # total assets
    s_tot_liab = _extract_row(bs_q, ["total liab", "totalliab", "totalliabilities"])  # liabilities
    s_equity = _extract_row(bs_q, ["total stockholder equity", "total stockholders equity", "totalshareholderequity", "totalstockholdersequity"])  # equity
    s_cur_assets = _extract_row(bs_q, ["total current assets", "totalcurrentassets"])  # current assets
    s_cur_liab = _extract_row(bs_q, ["total current liabilities", "totalcurrentliabilities"])  # current liabilities

    s_cfo = _extract_row(cf_q, ["total cash from operating activities", "operatingcashflow"])  # CFO
    s_capex = _extract_row(cf_q, ["capital expenditures", "capitalexpenditures"])  # CAPEX (usually negative)
    s_fcf = _extract_row(cf_q, ["free cash flow", "freecashflow"])  # FCF if available

    # Assemble DataFrame on union of period ends
    series_list: List[pd.Series] = [
        s_rev, s_gross, s_opinc, s_netinc, s_ebitda,
        s_tot_assets, s_tot_liab, s_equity, s_cur_assets, s_cur_liab,
        s_cfo, s_capex, s_fcf,
    ]
    valid = [s for s in series_list if s is not None and isinstance(s, pd.Series) and s.size > 0]

    # If quarterly is empty, try annual statements as a fallback
    if not valid and yf is not None:
        try:
            t = yf.Ticker(ticker)
            fin_a = _safe_df(getattr(t, "financials", None))
            bs_a = _safe_df(getattr(t, "balance_sheet", None))
            cf_a = _safe_df(getattr(t, "cashflow", None))
        except Exception:
            fin_a = bs_a = cf_a = None

        s_rev = _extract_row(fin_a, ["total revenue", "totalrevenue"]) if fin_a is not None else None
        s_gross = _extract_row(fin_a, ["gross profit", "grossprofit"]) if fin_a is not None else None
        s_opinc = _extract_row(fin_a, ["operating income", "operatingincome"]) if fin_a is not None else None
        s_netinc = _extract_row(fin_a, ["net income", "netincome"]) if fin_a is not None else None
        s_ebitda = _extract_row(fin_a, ["ebitda"]) if fin_a is not None else None

        s_tot_assets = _extract_row(bs_a, ["total assets", "totalassets"]) if bs_a is not None else None
        s_tot_liab = _extract_row(bs_a, ["total liab", "totalliab", "totalliabilities"]) if bs_a is not None else None
        s_equity = _extract_row(bs_a, ["total stockholder equity", "total stockholders equity", "totalshareholderequity", "totalstockholdersequity"]) if bs_a is not None else None
        s_cur_assets = _extract_row(bs_a, ["total current assets", "totalcurrentassets"]) if bs_a is not None else None
        s_cur_liab = _extract_row(bs_a, ["total current liabilities", "totalcurrentliabilities"]) if bs_a is not None else None

        s_cfo = _extract_row(cf_a, ["total cash from operating activities", "operatingcashflow"]) if cf_a is not None else None
        s_capex = _extract_row(cf_a, ["capital expenditures", "capitalexpenditures"]) if cf_a is not None else None
        s_fcf = _extract_row(cf_a, ["free cash flow", "freecashflow"]) if cf_a is not None else None

        series_list = [
            s_rev, s_gross, s_opinc, s_netinc, s_ebitda,
            s_tot_assets, s_tot_liab, s_equity, s_cur_assets, s_cur_liab,
            s_cfo, s_capex, s_fcf,
        ]
        valid = [s for s in series_list if s is not None and isinstance(s, pd.Series) and s.size > 0]

    if not valid:
        # Return empty aligned frame
        return pd.DataFrame(index=(align_index if align_index is not None else []))


    dfq = pd.concat(valid, axis=1)
    # Rename columns to canonical fundamental feature names (prefixed with f_)
    rename_map: Dict[str, str] = {
        "total revenue": "f_rev",
        "gross profit": "f_gross_profit",
        "operating income": "f_op_income",
        "net income": "f_net_income",
        "ebitda": "f_ebitda",
        "total assets": "f_assets",
        "total liab": "f_liab",
        "total liabilities": "f_liab",
        "total stockholder equity": "f_equity",
        "total stockholders equity": "f_equity",
        "totalshareholderequity": "f_equity",
        "totalstockholdersequity": "f_equity",
        "total current assets": "f_cur_assets",
        "total current liabilities": "f_cur_liab",
        "totalcurrentassets": "f_cur_assets",
        "totalcurrentliabilities": "f_cur_liab",
        "total cash from operating activities": "f_op_cf",
        "operatingcashflow": "f_op_cf",
        "capital expenditures": "f_capex",
        "capitalexpenditures": "f_capex",
        "free cash flow": "f_fcf",
        "freecashflow": "f_fcf",
    }

    cols = []
    for c in dfq.columns:
        nc = _norm_name(c)
        # Try exact mapping first based on original key (lowered, spaces kept in dict keys)
        mapped = None
        for k, v in rename_map.items():
            if _norm_name(k) == nc:
                mapped = v
                break
        cols.append(mapped or f"f_{nc}")
    dfq.columns = cols
    dfq = dfq.sort_index()

    # Derived ratios and growth (computed on quarterly series)
    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.replace(0, pd.NA)

    if "f_rev" in dfq.columns:
        if "f_gross_profit" in dfq.columns:
            dfq["f_gross_margin"] = _safe_div(dfq["f_gross_profit"], dfq["f_rev"]).astype(float)
        if "f_op_income" in dfq.columns:
            dfq["f_op_margin"] = _safe_div(dfq["f_op_income"], dfq["f_rev"]).astype(float)
        if "f_net_income" in dfq.columns:
            dfq["f_net_margin"] = _safe_div(dfq["f_net_income"], dfq["f_rev"]).astype(float)
        # YoY growth on quarterly series (change vs 4 quarters ago)
        dfq["f_rev_yoy"] = dfq["f_rev"].pct_change(4)

    if "f_net_income" in dfq.columns:
        dfq["f_netinc_yoy"] = dfq["f_net_income"].pct_change(4)

    if "f_liab" in dfq.columns and "f_equity" in dfq.columns:
        dfq["f_debt_to_equity"] = _safe_div(dfq["f_liab"], dfq["f_equity"]).astype(float)

    if "f_cur_assets" in dfq.columns and "f_cur_liab" in dfq.columns:
        dfq["f_current_ratio"] = _safe_div(dfq["f_cur_assets"], dfq["f_cur_liab"]).astype(float)

    # Free cash flow and margin
    if "f_fcf" not in dfq.columns:
        if "f_op_cf" in dfq.columns and "f_capex" in dfq.columns:
            # CAPEX often negative; fcf = cfo + capex
            dfq["f_fcf"] = (dfq["f_op_cf"].astype(float)) + (dfq["f_capex"].astype(float))
    if "f_fcf" in dfq.columns and "f_rev" in dfq.columns:
        dfq["f_fcf_margin"] = _safe_div(dfq["f_fcf"], dfq["f_rev"]).astype(float)

    # If alignment requested, forward-fill to the daily trading index
    if align_index is not None and len(align_index) > 0:
        # Reindex to daily dates (no backfill to avoid look-ahead)
        df_daily = dfq.reindex(pd.Index(sorted(dfq.index.union(align_index))))
        df_daily = df_daily.sort_index().ffill()
        df_daily = df_daily.reindex(align_index)
        return df_daily

    return dfq


def get_event_countdowns(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build known-future event countdown features aligned to a trading index.

    Returns a DataFrame with columns (all prefixed with 'f_' so they behave
    like static/fundamental features in the rest of the pipeline):
    - f_days_to_earnings
    - f_days_to_conference_call
    - f_days_to_ex_dividend
    - f_days_to_futures_expiry
    """
    if index is None or len(index) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    idx = pd.DatetimeIndex(index).sort_values()

    def _days_to_next_event(event_dates: List[pd.Timestamp]) -> pd.Series:
        """Return days until the next event for each trading date."""
        if not event_dates:
            return pd.Series(np.nan, index=idx, dtype="float32")
        # Ensure pure NumPy datetime64[D] arrays to avoid Timestamp/int
        # comparison issues inside np.searchsorted.
        ev = pd.to_datetime(event_dates).values.astype("datetime64[D]")
        idx_d = idx.values.astype("datetime64[D]")
        pos = np.searchsorted(ev, idx_d, side="left")
        days = np.full(len(idx_d), np.nan, dtype="float32")
        in_range = pos < len(ev)
        if np.any(in_range):
            diff = (ev[pos[in_range]] - idx_d[in_range]).astype("timedelta64[D]").astype("float32")
            days[in_range] = diff
        return pd.Series(days, index=idx, dtype="float32")

    earnings_dates: List[pd.Timestamp] = []
    ex_div_dates: List[pd.Timestamp] = []

    def _try_fetch_twse_events(tw_ticker: str) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """Best-effort TWSE event fetch. Returns (earnings, ex-div) lists or empty on failure."""
        try:
            # Strip .TW suffix for TWSE symbol
            sym = tw_ticker.replace(".TW", "").replace("TW", "")
            # TWSE endpoints are not reliably documented; placeholder for future expansion.
            # For now, return empty lists without logging noisy warnings.
            return [], []
        except Exception:
            return [], []

    # Prefer TWSE-specific hook for .TW tickers; skip yfinance to avoid noisy warnings.
    skip_yf = False
    if ticker.upper().endswith(".TW"):
        skip_yf = True
        ed_tw, div_tw = _try_fetch_twse_events(ticker)
        earnings_dates = ed_tw or earnings_dates
        ex_div_dates = div_tw or ex_div_dates

    if yf is not None and not skip_yf and (not earnings_dates or not ex_div_dates):
        try:
            t = yf.Ticker(ticker)
            if not earnings_dates:
                try:
                    ed = t.get_earnings_dates(limit=64)
                    if isinstance(ed, pd.DataFrame) and not ed.empty:
                        earnings_dates = list(pd.to_datetime(ed.index))
                except Exception:
                    ed = getattr(t, "earnings_dates", None)
                    if isinstance(ed, pd.DataFrame) and not ed.empty:
                        earnings_dates = list(pd.to_datetime(ed.index))
            if not ex_div_dates:
                try:
                    div = t.dividends
                    if isinstance(div, pd.Series) and not div.empty:
                        ex_div_dates = list(pd.to_datetime(div.index))
                except Exception:
                    pass
        except Exception:
            pass

    days_to_earnings = _days_to_next_event(earnings_dates)
    days_to_conf = days_to_earnings.copy()
    days_to_ex_div = _days_to_next_event(ex_div_dates)

    def _third_wednesdays(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        dates: List[pd.Timestamp] = []
        cur = pd.Timestamp(year=start.year, month=start.month, day=1)
        while cur <= end:
            month_end = (cur + pd.offsets.MonthEnd(0))
            days = pd.date_range(cur, month_end, freq="D")
            weds = [d for d in days if d.weekday() == 2]
            if len(weds) >= 3:
                dates.append(weds[2])
            cur = month_end + pd.Timedelta(days=1)
        return dates

    fut_expiry_dates = _third_wednesdays(idx.min(), idx.max() + pd.DateOffset(months=2))
    days_to_fut = _days_to_next_event(fut_expiry_dates)

    out = pd.DataFrame(
        {
            "f_days_to_earnings": days_to_earnings,
            "f_days_to_conference_call": days_to_conf,
            "f_days_to_ex_dividend": days_to_ex_div,
            "f_days_to_futures_expiry": days_to_fut,
        },
        index=idx,
    )
    return out.reindex(index)


def append_price_log(csv_path: str, price: float, source: str) -> None:
    """Append a single price datapoint to a CSV log."""
    now = dt.datetime.now()
    df = pd.DataFrame([[now, price, source]], columns=["time", "price", "source"])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


# -----------------------------
# VIX data retrieval utilities
# -----------------------------

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        last = df.columns.get_level_values(-1)
        try:
            df.columns = last
        except Exception:
            df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip("_") for tup in df.columns]
    return df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "volume": "Volume",
    })


def _get_alpha_vantage_key() -> Optional[str]:
    """Return Alpha Vantage API key from environment.

    Tries env var `ALPHAVANTAGE_API_KEY`; falls back to
    `secrets_local.ALPHAVANTAGE_API_KEY` if available.
    """
    return os.getenv("ALPHAVANTAGE_API_KEY") or _KEY_LOCAL


def _http_get_json(url: str, params: dict) -> Optional[dict]:
    """GET JSON using requests if available, else urllib. Returns dict or None."""
    try:
        if requests is not None:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            return None
        else:  # pragma: no cover
            from urllib.parse import urlencode
            from urllib.request import Request, urlopen

            full_url = url + ("?" + urlencode(params) if params else "")
            req = Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
    except Exception:
        return None


def _fetch_alpha_vantage_quote(symbol: str) -> Optional[float]:
    """Fetch last price via Alpha Vantage GLOBAL_QUOTE.

    Returns price as float or None on failure. Requires ALPHAVANTAGE_API_KEY.
    """
    api_key = _get_alpha_vantage_key()
    if not api_key:
        return None
    data = _http_get_json(
        "https://www.alphavantage.co/query",
        {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
    )
    try:
        if not data or "Global Quote" not in data:
            return None
        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price") or quote.get("05. Price")
        if price_str is None:
            return None
        return float(price_str)
    except Exception:
        return None


def _fetch_alpha_vantage_ohlcv(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV via Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.

    Returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    Filters by start/end. Returns None on failure or if API key missing.
    """
    api_key = _get_alpha_vantage_key()
    if not api_key:
        return None

    # Only use the provided symbol for consistency across sources
    symbols = [symbol]

    for sym in symbols:
        for func in ("TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_DAILY"):
            for outsz in ("full", "compact"):
                data = _http_get_json(
                    "https://www.alphavantage.co/query",
                    {
                        "function": func,
                        "symbol": sym,
                        "outputsize": outsz,
                        "apikey": api_key,
                    },
                )
                try:
                    ts = data.get("Time Series (Daily)") if data else None
                    if not ts:
                        continue
                    # Build DataFrame from time series
                    df = pd.DataFrame.from_dict(ts, orient="index")
                    # Map columns depending on endpoint
                    if func == "TIME_SERIES_DAILY_ADJUSTED":
                        df = df.rename(columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. adjusted close": "Adj Close",
                            "6. volume": "Volume",
                        })
                    else:
                        df = df.rename(columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. volume": "Volume",
                        })

                    # Ensure proper types
                    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    if "Volume" in df.columns:
                        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()

                    # Align to requested date range
                    start_dt = pd.to_datetime(start)
                    if end is not None:
                        end_dt = pd.to_datetime(end)
                        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
                    else:
                        df = df.loc[df.index >= start_dt]

                    # If no adjusted close, create it from Close
                    if "Adj Close" not in df.columns and "Close" in df.columns:
                        df["Adj Close"] = df["Close"]

                    # Apply the same adjustment factor to every price field.
                    # Replacing Close alone would create internally inconsistent
                    # OHLC bars and could fail High/Low envelope checks.
                    if auto_adjust and "Adj Close" in df.columns and "Close" in df.columns:
                        raw_close = df["Close"].replace(0, np.nan)
                        factor = df["Adj Close"] / raw_close
                        for price_col in ("Open", "High", "Low", "Close"):
                            if price_col in df.columns:
                                df[price_col] = df[price_col] * factor

                    # Reorder/ensure expected columns
                    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
                    df = df[cols]

                    if not df.empty:
                        return df.dropna(how="all")
                except Exception:
                    continue

    return None


def _load_cached_series(path: str) -> Optional[pd.Series]:
    try:
        if os.path.exists(path):
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = os.path.splitext(os.path.basename(path))[0]
            return s
    except Exception:
        return None
    return None


def _save_cached_series(path: str, series: pd.Series) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        series.to_frame(series.name or "value").to_csv(path)
    except Exception:
        pass


def _fetch_yf_series(ticker: str, start: str) -> Optional[pd.Series]:
    if yf is None:
        return None
    for attempt in ("download_start", "ticker_history", "download_period"):
        try:
            if attempt == "download_start":
                df = yf.download(ticker, start=start, auto_adjust=False, progress=False, group_by="column")
            elif attempt == "ticker_history":
                df = yf.Ticker(ticker).history(period="max", interval="1d")
                df.index.name = "Date"
            else:
                df = yf.download(ticker, period="max", auto_adjust=False, progress=False, group_by="column")
            df = _flatten_yf_columns(df)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            s = df.get("Close", df.get("Adj Close"))
            if s is not None and s.notna().sum() > 0:
                s.name = ticker
                return s
        except Exception:
            continue
    return None


def get_vix_data(start: str, align_index: Optional[pd.Index] = None) -> pd.DataFrame:
    """VIX functionality removed; returns empty DataFrame."""
    if align_index is not None:
        return pd.DataFrame(index=align_index)
    return pd.DataFrame()


def _http_get_bytes(url: str) -> Optional[bytes]:
    """GET raw bytes using requests or urllib. Returns bytes or None."""
    try:
        if requests is not None:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                return r.content
            return None
        else:  # pragma: no cover
            from urllib.request import Request, urlopen

            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=20) as resp:
                return resp.read()
    except Exception:
        return None


def _fetch_cboe_vix_series(kind: str) -> Optional[pd.Series]:
    """Fetch VIX or VIX3M series from CBOE official CSV.

    kind: 'vix' or 'vix3m'
    Returns a Series indexed by Date with name 'vix' or 'vix3m'.
    """
    base = "https://cdn.cboe.com/api/global/us_indices/daily_prices/"
    if kind.lower() == "vix":
        url = base + "VIX_History.csv"
        name = "vix"
    elif kind.lower() == "vix3m":
        url = base + "VIX3M_History.csv"
        name = "vix3m"
    else:
        return None

    raw = _http_get_bytes(url)
    if not raw:
        return None
    try:
        text = raw.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(text))
        # Normalize columns
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get("date") or cols_lower.get("date ")
        close_col = cols_lower.get("close") or cols_lower.get("closing")
        if date_col is None:
            # CBOE sometimes uses 'DATE'
            date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
        if close_col is None:
            close_col = next((c for c in df.columns if c.strip().lower() == "close"), None)
        if date_col is None or close_col is None:
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        s = pd.to_numeric(df[close_col], errors="coerce")
        s.name = name
        return s
    except Exception:
        return None

