"""
Bid-Ask Spread Computation and Analysis.

This module consolidates all spread-related computations from the IRP platform.

Definitions:
    Raw Spread = ask - bid
        AUTHOR MUST DEFINE FORMALLY: exact handling of missing/zero values,
        treatment of crossed quotes (bid > ask)

    Midpoint = (bid + ask) / 2
        AUTHOR MUST DEFINE FORMALLY: treatment when bid > ask (crossed quotes)

    Percentage Spread = (ask - bid) / midpoint
        AUTHOR MUST DEFINE FORMALLY: units (decimal vs percentage),
        edge case handling when midpoint = 0

    Basis Points Spread = spread_pct * 10000
        AUTHOR MUST DEFINE FORMALLY: confirm standard BPS definition

    Effective Spread = 2 * |trade_price - midpoint|
        AUTHOR MUST DEFINE FORMALLY: trade price source (last trade vs VWAP),
        time alignment between trade and quote

    Spread Collapse Slope = regression coefficient of log(spread) over lifecycle
        AUTHOR MUST DEFINE FORMALLY: regression method (OLS vs Theil-Sen),
        log transform justification, minimum observation threshold

References:
    - Section 5: Spread collapse slope computation
    - Section 6: Liquidity resilience via spread-shock recovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("microstructure.spread")


# Spread feature extraction
# Moved from: src/features/feature_engineering.py (SpreadFeatures class)

def compute_spread_features(
    df: pd.DataFrame,
    basis_points: bool = True,
    bid_candidates: list[str] | None = None,
    ask_candidates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract spread features from bid-ask data.

    Definition:
        spread = ask - bid
        midpoint = (bid + ask) / 2
        spread_pct = spread / midpoint
        spread_bps = spread_pct * 10000
        AUTHOR MUST DEFINE FORMALLY: handling of zero/negative spreads

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bid/ask data.
    basis_points : bool
        Whether to compute spread in basis points.
    bid_candidates : list[str] | None
        Column names to search for bid price.
    ask_candidates : list[str] | None
        Column names to search for ask price.

    Returns
    -------
    pd.DataFrame
        DataFrame with spread features added.

    Moved from
    ----------
    src/features/feature_engineering.py:287-330 (SpreadFeatures.extract)
    """
    df = df.copy()

    if bid_candidates is None:
        bid_candidates = ["yes_bid_c", "bid", "bid_price"]
    if ask_candidates is None:
        ask_candidates = ["yes_ask_c", "ask", "ask_price"]

    bid_col = _find_column(df, bid_candidates)
    ask_col = _find_column(df, ask_candidates)

    if bid_col is None or ask_col is None:
        return df

    bid = df[bid_col]
    ask = df[ask_col]

    df["spread"] = ask - bid
    df["midpoint"] = (bid + ask) / 2

    df["spread_pct"] = np.where(
        df["midpoint"] > 0,
        df["spread"] / df["midpoint"],
        np.nan,
    )

    if basis_points:
        df["spread_bps"] = df["spread_pct"] * 10000

    df["spread_ma"] = df["spread"].rolling(window=20, min_periods=1).mean()
    df["spread_tightening"] = df["spread"].diff()
    df["effective_spread"] = compute_effective_spread(df, bid_col, ask_col)

    return df


def compute_effective_spread(
    df: pd.DataFrame,
    bid_col: str,
    ask_col: str,
    price_candidates: list[str] | None = None,
) -> pd.Series:
    """
    Compute effective spread using trade prices.

    Definition:
        Effective Spread = 2 * |trade_price - midpoint|
        AUTHOR MUST DEFINE FORMALLY: trade price source and time alignment

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bid/ask and trade price data.
    bid_col : str
        Bid price column name.
    ask_col : str
        Ask price column name.
    price_candidates : list[str] | None
        Column names to search for trade price.

    Returns
    -------
    pd.Series
        Effective spread series.

    Moved from
    ----------
    src/features/feature_engineering.py:339-352 (SpreadFeatures._compute_effective_spread)
    """
    if price_candidates is None:
        price_candidates = ["price_c", "price", "close"]

    price_col = _find_column(df, price_candidates)
    if price_col is None:
        return pd.Series(np.nan, index=df.index)

    midpoint = (df[bid_col] + df[ask_col]) / 2
    return 2 * (df[price_col] - midpoint).abs()


# Spread dynamics analysis
# Moved from: src/microstructure/analysis.py (SpreadAnalyzer class)

def compute_spread_collapse_slope(
    df: pd.DataFrame,
    lifecycle_col: str = "lifecycle_ratio",
) -> dict[str, float]:
    """
    Fast slope estimate of log(spread) over lifecycle.

    Definition:
        Spread Collapse Slope = OLS coefficient of log(spread) ~ lifecycle_ratio
        AUTHOR MUST DEFINE FORMALLY: regression method, log transform justification,
        downsampling approach (N=200), minimum observation threshold (10)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread and lifecycle_ratio columns.
    lifecycle_col : str
        Lifecycle column name.

    Returns
    -------
    dict[str, float]
        Slope estimate.

    Moved from
    ----------
    src/microstructure/analysis.py:210-261 (SpreadAnalyzer.robust_spread_collapse_slope)
    """
    if "spread" not in df.columns or lifecycle_col not in df.columns:
        return {"spread_collapse_slope": np.nan}

    spread = df["spread"].to_numpy()
    lifecycle = df[lifecycle_col].to_numpy()

    valid = np.isfinite(spread) & np.isfinite(lifecycle) & (spread > 0)
    spread = spread[valid]
    lifecycle = lifecycle[valid]

    if len(spread) < 10:
        return {"spread_collapse_slope": np.nan}

    # Downsample to at most 200 points for speed
    N = 200
    if len(spread) > N:
        idx = np.linspace(0, len(spread) - 1, N).astype(int)
        spread = spread[idx]
        lifecycle = lifecycle[idx]

    log_spread = np.log(spread)

    den = np.var(lifecycle)
    if den == 0:
        return {"spread_collapse_slope": np.nan}

    num = np.cov(lifecycle, log_spread, bias=True)[0, 1]
    return {"spread_collapse_slope": num / den}


def compute_spread_curve(
    df: pd.DataFrame,
    resample_freq: str = "5min",
) -> pd.DataFrame:
    """
    Compute spread tightening curve.

    Definition:
        Spread curve = resampled spread statistics over time
        tightening = first - last (within window)
        AUTHOR MUST DEFINE FORMALLY: aggregation method, window selection

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread data.
    resample_freq : str
        Resampling frequency.

    Returns
    -------
    pd.DataFrame
        Spread curve statistics.

    Moved from
    ----------
    src/microstructure/analysis.py:341-373 (SpreadAnalyzer.compute_spread_curve)
    """
    if "spread" not in df.columns:
        return pd.DataFrame()

    resampled = df["spread"].resample(resample_freq).agg([
        "mean", "std", "min", "max", "first", "last"
    ])

    resampled["tightening"] = resampled["first"] - resampled["last"]
    resampled["tightening_pct"] = np.where(
        resampled["first"] > 0,
        resampled["tightening"] / resampled["first"],
        np.nan,
    )

    return resampled


def compute_spread_percentiles(
    df: pd.DataFrame,
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute spread percentile statistics.

    Definition:
        Spread percentiles = distribution quantiles of spread
        AUTHOR MUST DEFINE FORMALLY: percentile selection justification

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread data.
    percentiles : list[float] | None
        Percentiles to compute.

    Returns
    -------
    dict[str, float]
        Percentile values.

    Moved from
    ----------
    src/microstructure/analysis.py:375-405 (SpreadAnalyzer.compute_spread_percentiles)
    """
    if "spread" not in df.columns:
        return {}

    if percentiles is None:
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    spread = df["spread"].dropna()
    result = {}

    for p in percentiles:
        result[f"spread_p{int(p*100)}"] = spread.quantile(p)

    return result


def compute_spread_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread statistics by regime.

    Definition:
        Regime-conditional spread statistics
        AUTHOR MUST DEFINE FORMALLY: regime definitions (see regimes.py)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spread and regime data.

    Returns
    -------
    pd.DataFrame
        Spread statistics by regime.

    Moved from
    ----------
    src/microstructure/analysis.py:407-426 (SpreadAnalyzer.compute_spread_regime_stats)
    """
    if "spread" not in df.columns or "regime" not in df.columns:
        return pd.DataFrame()

    return df.groupby("regime")["spread"].agg([
        "mean", "std", "median", "count"
    ])


# Helper functions

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
