"""
Liquidity Depth and Resilience Metrics.

This module consolidates all liquidity-related computations including
volume features, depth analysis, and resilience measurement.

Definitions:
    Volume Z-Score:
        z_t = (V_t - μ_t) / σ_t  if σ_t > 0, else 0
        where:
            μ_t = rolling mean of volume over window W (default W=20)
            σ_t = rolling std of volume over window W
            min_periods = 1 (computation begins from first observation)
        Zero standard deviation yields z_t = 0 (not NaN).

    Book Thinning:
        book_thinning_t = (bid_range_t + ask_range_t) / 2
        where:
            bid_range_t = bid_high_t - bid_low_t
            ask_range_t = ask_high_t - ask_low_t
        Column priority: ["yes_bid_h", "bid_high"] for high, ["yes_bid_l", "bid_low"] for low.
        Units: same as input price units (typically cents for prediction markets).
        Interpretation: average intra-bar price movement on bid and ask sides;
        larger values indicate wider price oscillation within the bar.

    Depth Resilience:
        R_t = N_negative / (N_positive + N_negative)  over window W (default W=10)
        where:
            Δspread_t = spread_t - spread_{t-1}
            N_negative = count(Δspread < 0) in window
            N_positive = count(Δspread > 0) in window
        Returns NaN if window has fewer than 2 observations or all changes are zero.
        Range: [0, 1] where 1 = all spread changes were decreases (high resilience).

    Liquidity Resilience (Shock-Recovery):
        Shock event: spread_t > Q_{0.95}(spread) AND spread_{t-1} <= Q_{0.95}(spread)
        where Q_p denotes the p-th quantile computed over the full spread series.
        Median baseline: M = median(spread) computed over full series.
        Recovery: first timestamp t' in [t, t + 30 min] where spread_{t'} <= M.
        Recovery time: (t' - t) in minutes.
        Shocks without recovery within window are excluded from recovery time statistics.

    Recovery Rate:
        recovery_rate = |{shocks with recovery}| / |{all shocks}|
        A shock is considered recovered if any observation within recovery_window_minutes
        satisfies spread <= median_spread. If shock_count = 0, recovery_rate = 1.0.

References:
    - Section 6: Liquidity resilience via spread-shock recovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("microstructure.liquidity")


# Volume features
# Moved from: src/features/feature_engineering.py (LiquidityFeatures class)

def compute_volume_features(
    df: pd.DataFrame,
    volume_ma_window: int = 20,
) -> pd.DataFrame:
    """
    Extract volume-based liquidity features.

    Definition:
        volume_ma_t = (1/W) * Σ_{i=t-W+1}^{t} V_i  (rolling mean, min_periods=1)
        volume_std_t = sqrt((1/(W-1)) * Σ_{i=t-W+1}^{t} (V_i - volume_ma_t)²)  (rolling std, min_periods=1)
        volume_zscore_t = (V_t - volume_ma_t) / volume_std_t  if volume_std_t > 0, else 0
        cumulative_volume_t = Σ_{i=1}^{t} V_i
        volume_surge_t = V_t / volume_ma_t  (NaN if volume_ma_t = 0)
        avg_trade_size_t = V_t / trade_count_t  if trade_count_t > 0, else NaN
        log_volume_t = log(1 + V_t)

        Window W = volume_ma_window parameter (default 20).
        min_periods = 1 for all rolling computations.
        Zero volume observations are included in computations; they are not filtered.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume data.
    volume_ma_window : int
        Window size for moving average.

    Returns
    -------
    pd.DataFrame
        DataFrame with volume features added.

    Moved from
    ----------
    src/features/feature_engineering.py:224-270 (LiquidityFeatures.extract)
    """
    df = df.copy()

    if "volume" not in df.columns:
        return df

    df["volume_ma"] = df["volume"].rolling(
        window=volume_ma_window, min_periods=1
    ).mean()

    df["volume_std"] = df["volume"].rolling(
        window=volume_ma_window, min_periods=1
    ).std()

    df["volume_zscore"] = np.where(
        df["volume_std"] > 0,
        (df["volume"] - df["volume_ma"]) / df["volume_std"],
        0,
    )

    df["cumulative_volume"] = df["volume"].cumsum()
    df["volume_surge"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

    if "trade_count" in df.columns:
        df["avg_trade_size"] = np.where(
            df["trade_count"] > 0,
            df["volume"] / df["trade_count"],
            np.nan,
        )

    df["log_volume"] = np.log1p(df["volume"])

    return df


# Depth features
# Moved from: src/features/feature_engineering.py (DepthFeatures class)

def compute_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract order book depth features.

    Definition:
        bid_range_t = bid_high_t - bid_low_t
        ask_range_t = ask_high_t - ask_low_t
        book_thinning_t = (bid_range_t + ask_range_t) / 2

        Column resolution (first match used):
            bid_high: ["yes_bid_h", "bid_high"]
            bid_low: ["yes_bid_l", "bid_low"]
            ask_high: ["yes_ask_h", "ask_high"]
            ask_low: ["yes_ask_l", "ask_low"]

        Interpretation: bid_range and ask_range represent the intra-bar price
        range on each side of the order book (high minus low within the bar).
        book_thinning is the average of these ranges. This is a proxy for
        quote volatility within the bar, not actual order book depth levels.
        Higher values indicate more price movement within the aggregation period.

        If spread column exists, depth_resilience is also computed via
        compute_depth_resilience(df).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC bid/ask data.

    Returns
    -------
    pd.DataFrame
        DataFrame with depth features added.

    Moved from
    ----------
    src/features/feature_engineering.py:366-399 (DepthFeatures.extract)
    """
    df = df.copy()

    bid_h = _find_column(df, ["yes_bid_h", "bid_high"])
    bid_l = _find_column(df, ["yes_bid_l", "bid_low"])
    ask_h = _find_column(df, ["yes_ask_h", "ask_high"])
    ask_l = _find_column(df, ["yes_ask_l", "ask_low"])

    if bid_h and bid_l:
        df["bid_range"] = df[bid_h] - df[bid_l]

    if ask_h and ask_l:
        df["ask_range"] = df[ask_h] - df[ask_l]

    if bid_h and bid_l and ask_h and ask_l:
        df["book_thinning"] = (df["bid_range"] + df["ask_range"]) / 2

    if "spread" in df.columns:
        df["depth_resilience"] = compute_depth_resilience(df)

    return df


# Depth resilience
# Moved from: src/microstructure/analysis.py (DepthAnalyzer class)

def compute_depth_resilience(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """
    Compute depth resilience metric.

    Definition:
        Δspread_t = spread_t - spread_{t-1}
        For each rolling window of size W (default 10):
            N_positive = |{i : Δspread_i > 0}|  (spread increases)
            N_negative = |{i : Δspread_i < 0}|  (spread decreases)
            R_t = N_negative / (N_positive + N_negative)

        Edge cases:
            - If window contains fewer than 2 observations: R_t = NaN
            - If N_positive + N_negative = 0 (all changes are exactly zero): R_t = NaN
            - If spread column missing: returns Series of NaN

        Range: [0, 1]
        Interpretation: proportion of spread changes that were decreases (tightening).
        Higher values indicate the spread tends to decrease more often than increase,
        suggesting faster recovery from widening. This is a statistical proxy for
        resilience, not a direct measure of order book depth recovery.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread data.
    window : int
        Analysis window.

    Returns
    -------
    pd.Series
        Depth resilience scores (0 to 1).

    Moved from
    ----------
    src/microstructure/analysis.py:575-610 (DepthAnalyzer.compute_depth_resilience)
    src/features/feature_engineering.py:408-419 (DepthFeatures._compute_depth_resilience)
    """
    if "spread" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    spread_change = df["spread"].diff()

    def recovery_rate(x):
        if len(x) < 2:
            return np.nan
        positive = (x > 0).sum()
        negative = (x < 0).sum()
        total = positive + negative
        if total == 0:
            return np.nan
        return negative / total

    return spread_change.rolling(window=window).apply(recovery_rate, raw=False)


def compute_depth_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume-weighted depth impact.

    Definition:
        volume_ma_t = rolling mean of volume over 20 periods (min_periods=1)
        volume_spread_corr_t = Pearson correlation of (V, spread) over rolling window of 60 periods
        large_volume_mask_t = True if V_t > 2 * volume_ma_t

        post_large_spread_t = spread_t if large_volume_mask_{t-1} = True, else NaN

        Interpretation:
            volume_spread_corr: measures contemporaneous relationship between
            trading activity and spread width. Positive correlation suggests
            higher volume widens spreads (liquidity consumption).

            post_large_spread: captures the spread observed immediately after
            a large volume event (defined as volume exceeding 2x its 20-period MA).
            Used to assess whether large trades impact subsequent liquidity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume and spread data.

    Returns
    -------
    pd.DataFrame
        Depth impact metrics.

    Moved from
    ----------
    src/microstructure/analysis.py:612-645 (DepthAnalyzer.compute_depth_impact)
    """
    if "volume" not in df.columns or "spread" not in df.columns:
        return pd.DataFrame()

    result = pd.DataFrame(index=df.index)

    volume_ma = df["volume"].rolling(window=20, min_periods=1).mean()

    result["volume_spread_corr"] = df["volume"].rolling(window=60).corr(df["spread"])

    large_volume_mask = df["volume"] > 2 * volume_ma
    result["post_large_spread"] = np.where(
        large_volume_mask.shift(1).fillna(False),
        df["spread"],
        np.nan,
    )

    return result


# Liquidity resilience (shock-recovery)
# Moved from: src/microstructure/analysis.py (SpreadAnalyzer class)

def compute_liquidity_resilience(
    df: pd.DataFrame,
    shock_threshold_quantile: float = 0.95,
    recovery_window_minutes: int = 30,
) -> dict[str, float]:
    """
    Compute liquidity resilience via spread-shock recovery time.

    Definition:
        Let S = {spread_t} be the spread time series.
        Q_p = p-th quantile of S (computed over full series)
        M = median(S) (computed over full series)

        Shock threshold: τ = Q_{shock_threshold_quantile} (default Q_{0.95})

        Shock onset detection:
            shock_t = True if (spread_t > τ) AND (spread_{t-1} <= τ)
            This identifies transitions into the shocked state, not all shocked observations.

        Recovery detection:
            For each shock onset at time t, examine window [t, t + recovery_window_minutes].
            Recovery occurs at first t' in window where spread_{t'} <= M.
            Recovery time = (t' - t) converted to minutes.
            If no such t' exists within window, shock is considered unrecovered.

        Output metrics:
            shock_count = total number of shock onset events
            median_recovery_minutes = median of recovery times (excluding unrecovered)
            mean_recovery_minutes = mean of recovery times (excluding unrecovered)
            recovery_rate = (# recovered shocks) / shock_count

        Edge cases:
            - If len(df) < 20 or spread column missing: returns zeros/NaN
            - If shock_count = 0: recovery_rate = 1.0 (perfect resilience by convention)
            - Shocks with < 2 observations in window are skipped

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread and DatetimeIndex.
    shock_threshold_quantile : float
        Quantile threshold for shock detection (default 0.95).
    recovery_window_minutes : int
        Maximum window to measure recovery (default 30).

    Returns
    -------
    dict[str, float]
        Recovery metrics:
        - shock_count: Number of detected shock events
        - median_recovery_minutes: Median time to recover
        - mean_recovery_minutes: Mean recovery time
        - recovery_rate: Proportion of shocks that recovered (0-1)

    Moved from
    ----------
    src/microstructure/analysis.py:263-339 (SpreadAnalyzer.compute_liquidity_resilience)
    """
    if "spread" not in df.columns or len(df) < 20:
        return {
            "shock_count": 0,
            "median_recovery_minutes": np.nan,
            "mean_recovery_minutes": np.nan,
            "recovery_rate": np.nan,
        }

    spread = df["spread"]
    median_spread = spread.median()
    shock_threshold = spread.quantile(shock_threshold_quantile)

    # Identify shock events
    shock_mask = spread > shock_threshold
    shock_starts = shock_mask & ~shock_mask.shift(1, fill_value=False)
    shock_indices = df.index[shock_starts]

    if len(shock_indices) == 0:
        return {
            "shock_count": 0,
            "median_recovery_minutes": np.nan,
            "mean_recovery_minutes": np.nan,
            "recovery_rate": 1.0,  # No shocks = perfect resilience
        }

    recovery_times = []

    for shock_time in shock_indices:
        window_end = shock_time + pd.Timedelta(minutes=recovery_window_minutes)
        window_data = spread.loc[shock_time:window_end]

        if len(window_data) < 2:
            continue

        recovered_mask = window_data <= median_spread
        if recovered_mask.any():
            recovery_idx = window_data.index[recovered_mask][0]
            recovery_minutes = (recovery_idx - shock_time).total_seconds() / 60
            recovery_times.append(recovery_minutes)

    shock_count = len(shock_indices)
    recovery_rate = len(recovery_times) / shock_count if shock_count > 0 else np.nan

    return {
        "shock_count": shock_count,
        "median_recovery_minutes": np.median(recovery_times) if recovery_times else np.nan,
        "mean_recovery_minutes": np.mean(recovery_times) if recovery_times else np.nan,
        "recovery_rate": recovery_rate,
    }


# Helper functions

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
