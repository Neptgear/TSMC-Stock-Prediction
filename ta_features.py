import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14, use_ema: bool = True) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    if use_ema:
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    denom = (upper - lower).replace(0, np.nan)
    width = (upper - lower) / (mid.replace(0, np.nan))
    percent_b = (close - lower) / denom
    return mid, upper, lower, width, percent_b


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period)


# ----------------------
# Additional indicators
# ----------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
    up_idx = high.rolling(period).apply(lambda x: np.argmax(x), raw=True)
    down_idx = low.rolling(period).apply(lambda x: np.argmin(x), raw=True)
    up_since = period - 1 - up_idx
    down_since = period - 1 - down_idx
    aroon_up = 100 * (period - up_since) / period
    aroon_down = 100 * (period - down_since) / period
    return aroon_up, aroon_down


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume.fillna(0)
    delta_tp = tp.diff()
    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume.fillna(0)
    return mfv.rolling(period).sum() / volume.fillna(0).rolling(period).sum().replace(0, np.nan)


def rolling_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    pv = tp * volume.fillna(0)
    return pv.rolling(period).sum() / volume.fillna(0).rolling(period).sum().replace(0, np.nan)


def compute_features(
    df: pd.DataFrame,
    target: str = "next_close",
    horizon: int = 1,
    include_extra: bool = True,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute technical indicators and target column.

    Parameters:
    - df: DataFrame with columns Open, High, Low, Close, Adj Close, Volume
    - target: 'next_close' or 'next_return_pct'
    - horizon: steps ahead for the target (default 1 day)
    - include_extra: if True, add additional indicators beyond basics

    Returns: DataFrame with feature columns and 'target'.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data = df.copy().sort_index()

    # Calendar/time features – known in advance (no look-ahead).
    idx = data.index
    if isinstance(idx, pd.DatetimeIndex):
        data["year"] = idx.year
        data["month"] = idx.month
        data["day_of_week"] = idx.dayofweek
        data["is_month_end"] = idx.is_month_end.astype(int)
        data["is_quarter_end"] = idx.is_quarter_end.astype(int)
        idx_series = pd.Series(idx)
        next_day = idx_series.shift(-1)
        gap = (next_day - idx_series).dt.days
        data["is_holiday_eve"] = (gap > 1).astype(int).fillna(0)

    # Known-future event countdown features (if provided)
    if events_df is not None and not events_df.empty:
        overlap = [c for c in events_df.columns if c in data.columns]
        if overlap:
            events_df = events_df.drop(columns=overlap)
        data = data.join(events_df, how="left")

    # Moving averages
    data["ma5"] = data["Close"].rolling(5).mean()
    data["ma10"] = data["Close"].rolling(10).mean()
    data["ma20"] = data["Close"].rolling(20).mean()
    data["ma50"] = data["Close"].rolling(50).mean()
    data["ma100"] = data["Close"].rolling(100).mean()
    data["ma200"] = data["Close"].rolling(200).mean()
    # EMAs
    data["ema12"] = _ema(data["Close"], 12)
    data["ema26"] = _ema(data["Close"], 26)

    # RSI
    data["rsi14"] = rsi(data["Close"], period=14)

    # MACD
    macd_line, signal_line, hist = macd(data["Close"], 12, 26, 9)
    data["macd"] = macd_line
    data["macd_signal"] = signal_line
    data["macd_hist"] = hist

    # Bollinger Bands
    mid, upper, lower, width, pb = bollinger_bands(data["Close"], 20, 2.0)
    data["bb_mid"] = mid
    data["bb_upper"] = upper
    data["bb_lower"] = lower
    data["bb_width"] = width
    data["bb_percent_b"] = pb

    # Volume change rate (guard against division by zero -> inf)
    data["vol_chg"] = data["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    # Multi-horizon returns and realized volatility
    data["ret_1"] = data["Close"].pct_change(1)
    data["ret_5"] = data["Close"].pct_change(5)
    data["ret_10"] = data["Close"].pct_change(10)
    data["ret_20"] = data["Close"].pct_change(20)
    data["vol20"] = data["ret_1"].rolling(20).std()
    data["vol60"] = data["ret_1"].rolling(60).std()

    # Price location relative to trend
    for w in (20, 50, 200):
        ma = data[f"ma{w}"]
        data[f"close_ma{w}_ratio"] = (data["Close"] / ma) - 1.0

    # Z-score of price and drawdown
    data["zscore20"] = (data["Close"] - data["Close"].rolling(20).mean()) / data["Close"].rolling(20).std(ddof=0)
    data["zscore60"] = (data["Close"] - data["Close"].rolling(60).mean()) / data["Close"].rolling(60).std(ddof=0)
    roll_max_252 = data["Close"].rolling(252).max()
    data["drawdown252"] = (data["Close"] / roll_max_252) - 1.0

    if include_extra:
        # Stochastic Oscillator
        k, d = stochastic_kd(data["High"], data["Low"], data["Close"], 14, 3)
        data["stoch_k"] = k
        data["stoch_d"] = d

        # ATR
        data["atr14"] = atr(data["High"], data["Low"], data["Close"], 14)

        # OBV
        data["obv"] = obv(data["Close"], data["Volume"]).fillna(0)

        # Rate of change
        data["roc10"] = roc(data["Close"], 10)
        data["roc20"] = roc(data["Close"], 20)

        # ADX & DI
        adx14, pdi14, mdi14 = adx(data["High"], data["Low"], data["Close"], 14)
        data["adx14"] = adx14
        data["plus_di14"] = pdi14
        data["minus_di14"] = mdi14

        # Aroon
        aru, ard = aroon(data["High"], data["Low"], 25)
        data["aroon_up25"] = aru
        data["aroon_down25"] = ard

        # CCI, Williams %R, MFI, CMF
        data["cci20"] = cci(data["High"], data["Low"], data["Close"], 20)
        data["willr14"] = williams_r(data["High"], data["Low"], data["Close"], 14)
        data["mfi14"] = mfi(data["High"], data["Low"], data["Close"], data["Volume"], 14)
        data["cmf20"] = cmf(data["High"], data["Low"], data["Close"], data["Volume"], 20)

        # Rolling VWAP and ratio
        data["vwap20"] = rolling_vwap(data["High"], data["Low"], data["Close"], data["Volume"], 20)
        data["vwap_ratio20"] = (data["Close"] / data["vwap20"]) - 1.0

    # Target construction
    if target == "next_close":
        data["target"] = data["Close"].shift(-horizon)
    elif target in ("next_return_pct", "next_ret_pct"):
        future_close = data["Close"].shift(-horizon)
        data["target"] = (future_close - data["Close"]) / data["Close"]
    else:
        raise ValueError("target must be 'next_close' or 'next_return_pct'")

    # Replace infinities globally, but only require the target to be non-NaN.
    # Other columns (including fundamentals and event countdowns) are allowed
    # to contain NaNs and are filtered later per-feature in the training code.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target"]).copy()
    return data


def compute_indicators_only(
    df: pd.DataFrame,
    include_extra: bool = True,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute indicators without constructing a target column.

    Keeps the latest rows (no horizon-based trimming). Drops rows with
    NaNs introduced by indicator warm-up but preserves the tail of the series.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data = df.copy().sort_index()

    # Calendar/time features – known in advance (no look-ahead).
    idx = data.index
    if isinstance(idx, pd.DatetimeIndex):
        data["year"] = idx.year
        data["month"] = idx.month
        data["day_of_week"] = idx.dayofweek
        data["is_month_end"] = idx.is_month_end.astype(int)
        data["is_quarter_end"] = idx.is_quarter_end.astype(int)
        idx_series = pd.Series(idx)
        next_day = idx_series.shift(-1)
        gap = (next_day - idx_series).dt.days
        data["is_holiday_eve"] = (gap > 1).astype(int).fillna(0)

    data["ma5"] = data["Close"].rolling(5).mean()
    data["ma10"] = data["Close"].rolling(10).mean()
    data["ma20"] = data["Close"].rolling(20).mean()
    data["ma50"] = data["Close"].rolling(50).mean()
    data["ma100"] = data["Close"].rolling(100).mean()
    data["ma200"] = data["Close"].rolling(200).mean()
    data["ema12"] = _ema(data["Close"], 12)
    data["ema26"] = _ema(data["Close"], 26)

    data["rsi14"] = rsi(data["Close"], period=14)
    macd_line, signal_line, hist = macd(data["Close"], 12, 26, 9)
    data["macd"] = macd_line
    data["macd_signal"] = signal_line
    data["macd_hist"] = hist

    mid, upper, lower, width, pb = bollinger_bands(data["Close"], 20, 2.0)
    data["bb_mid"] = mid
    data["bb_upper"] = upper
    data["bb_lower"] = lower
    data["bb_width"] = width
    data["bb_percent_b"] = pb

    data["vol_chg"] = data["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    data["ret_1"] = data["Close"].pct_change(1)
    data["ret_5"] = data["Close"].pct_change(5)
    data["ret_10"] = data["Close"].pct_change(10)
    data["ret_20"] = data["Close"].pct_change(20)
    data["vol20"] = data["ret_1"].rolling(20).std()
    data["vol60"] = data["ret_1"].rolling(60).std()

    for w in (20, 50, 200):
        ma = data[f"ma{w}"]
        data[f"close_ma{w}_ratio"] = (data["Close"] / ma) - 1.0

    data["zscore20"] = (data["Close"] - data["Close"].rolling(20).mean()) / data["Close"].rolling(20).std(ddof=0)
    data["zscore60"] = (data["Close"] - data["Close"].rolling(60).mean()) / data["Close"].rolling(60).std(ddof=0)
    roll_max_252 = data["Close"].rolling(252).max()
    data["drawdown252"] = (data["Close"] / roll_max_252) - 1.0

    if include_extra:
        k, d = stochastic_kd(data["High"], data["Low"], data["Close"], 14, 3)
        data["stoch_k"] = k
        data["stoch_d"] = d
        data["atr14"] = atr(data["High"], data["Low"], data["Close"], 14)
        data["obv"] = obv(data["Close"], data["Volume"]).fillna(0)
        data["roc10"] = roc(data["Close"], 10)
        data["roc20"] = roc(data["Close"], 20)
        adx14, pdi14, mdi14 = adx(data["High"], data["Low"], data["Close"], 14)
        data["adx14"] = adx14
        data["plus_di14"] = pdi14
        data["minus_di14"] = mdi14
        aru, ard = aroon(data["High"], data["Low"], 25)
        data["aroon_up25"] = aru
        data["aroon_down25"] = ard
        data["cci20"] = cci(data["High"], data["Low"], data["Close"], 20)
        data["willr14"] = williams_r(data["High"], data["Low"], data["Close"], 14)
        data["mfi14"] = mfi(data["High"], data["Low"], data["Close"], data["Volume"], 14)
        data["cmf20"] = cmf(data["High"], data["Low"], data["Close"], data["Volume"], 20)
        data["vwap20"] = rolling_vwap(data["High"], data["Low"], data["Close"], data["Volume"], 20)
        data["vwap_ratio20"] = (data["Close"] / data["vwap20"]) - 1.0

    # For charting we only need the essential columns clean; allow NaNs in
    # indicator warm-up tails so we don't drop the entire history.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Close"]).copy()
    return data


def get_default_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "year", "month", "day_of_week", "is_month_end", "is_quarter_end",
        "Open", "High", "Low", "Close", "Volume",
        "ma5", "ma10", "ma20", "ma50", "ma100", "ma200",
        "ema12", "ema26",
        "rsi14",
        "macd", "macd_signal", "macd_hist",
        "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_percent_b",
        "vol_chg",
        "stoch_k", "stoch_d",
        "atr14",
        "obv",
        "roc10", "roc20",
        "ret_1", "ret_5", "ret_10", "ret_20",
        "vol20", "vol60",
        "close_ma20_ratio", "close_ma50_ratio", "close_ma200_ratio",
        "zscore20", "zscore60",
        "drawdown252",
        "adx14", "plus_di14", "minus_di14",
        "aroon_up25", "aroon_down25",
        "cci20", "willr14", "mfi14",
        "vwap20", "vwap_ratio20",
    ]
    tech = [c for c in candidates if c in df.columns]
    # Append any available fundamental columns (prefixed with 'f_')
    fund = [c for c in df.columns if isinstance(c, str) and c.startswith("f_")]
    # Keep a consistent order for fundamentals
    fund.sort()
    return tech + fund


def get_tft_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Curated feature list tailored for the lightweight TFT-style model.

    Roughly follows the "Known Future / Observed Past / Static" guidance:
    - Known future: calendar/time features that are fixed in advance.
    - Observed past: price/volume/indicator history.
    - Static: fundamentals (f_*) treated as per-window static context.
    """
    known_future = [
        "year",
        "month",
        "day_of_week",
        "is_month_end",
        "is_quarter_end",
        # Note: event countdowns (f_days_*) are treated as static/fundamental.
    ]
    observed_past = [
        "Open", "High", "Low", "Close", "Volume",
        "atr14",
        "ma5", "ma10", "ma20", "ma50", "ma100",
        "ema12", "ema26",
        "rsi14",
        "stoch_k", "stoch_d",
        "macd", "macd_signal", "macd_hist",
        "bb_mid", "bb_upper", "bb_lower", "bb_percent_b",
        "vol20", "vol60",
        "zscore20", "zscore60",
        "adx14", "plus_di14", "minus_di14",
    ]
    kf_cols = [c for c in known_future if c in df.columns]
    op_cols = [c for c in observed_past if c in df.columns]
    fund_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("f_")]
    fund_cols.sort()
    return kf_cols + op_cols + fund_cols


def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature DataFrame into sliding windows for sequence models.

    Returns:
    - X: shape (n_samples, window_size, n_features)
    - y: shape (n_samples,)
    """
    values = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    n_total = len(df)
    n_features = len(feature_cols)
    if n_total <= window_size:
        raise ValueError("Not enough rows for the given window_size")

    xs, ys = [], []
    # Build samples ending at end_idx-1; do not include extra tail row
    for end_idx in range(window_size, n_total + 1):
        # Window covers [end_idx - window_size, end_idx - 1]
        start_idx = end_idx - window_size
        xs.append(values[start_idx:end_idx])
        # Label corresponds to target at anchor time end_idx - 1 (t+h)
        ys.append(targets[end_idx - 1])
    X = np.stack(xs, axis=0).reshape(-1, window_size, n_features)
    y = np.array(ys).reshape(-1)
    return X, y


def make_sliding_windows_with_base(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    base_col: str = "Close",
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like make_sliding_windows, but also returns a per-sample baseline value
    (e.g., last observed Close before prediction) to enable residual training.

    Returns:
    - X: (n_samples, T, F)
    - y: (n_samples,)
    - base: (n_samples,) where base[i] = df[base_col][end_idx-1]
    """
    values = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    base_series = df[base_col].values.astype(np.float32)

    n_total = len(df)
    n_features = len(feature_cols)
    if n_total <= window_size:
        raise ValueError("Not enough rows for the given window_size")

    xs, ys, bs = [], [], []
    for end_idx in range(window_size, n_total + 1):
        start_idx = end_idx - window_size
        xs.append(values[start_idx:end_idx])
        # Label at anchor (end_idx - 1)
        ys.append(targets[end_idx - 1])
        # Baseline is the last observed value within window (end_idx - 1)
        bs.append(base_series[end_idx - 1])
    X = np.stack(xs, axis=0).reshape(-1, window_size, n_features)
    y = np.array(ys).reshape(-1)
    b = np.array(bs).reshape(-1)
    return X, y, b


def time_series_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    split = int(n * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]
