"""
Bid-Ask Spread Computation and Analysis.

This module consolidates all spread-related computations from the IRP platform.

Definitions:
    Raw Spread = ask - bid
        Missing values propagate as NaN. Crossed quotes (bid > ask) are not
        special-cased and will produce negative spreads.

    Midpoint = (bid + ask) / 2
        Crossed quotes produce valid midpoints between bid and ask.

    Percentage Spread = (ask - bid) / midpoint
        Returns decimal values (e.g., 0.05 for 5%). Returns NaN when midpoint <= 0.

    Basis Points Spread = spread_pct * 10000
        Standard basis points conversion (1 bp = 0.01%).

    Effective Spread = 2 * |trade_price - midpoint|
        Uses last trade price aligned with contemporaneous quote.

    Spread Collapse Slope = OLS coefficient of log(spread) over lifecycle
        Log transform captures proportional spread tightening. Requires minimum
        10 observations. Downsampled to 200 points for computational efficiency.

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
        spread_pct = spread / midpoint (NaN when midpoint <= 0)
        spread_bps = spread_pct * 10000

        Zero or negative spreads are preserved; filtering should be done
        downstream if needed.

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

        Uses last trade price (price_c/price/close) aligned with
        contemporaneous bid-ask quote.

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

        Uses simple OLS regression. Log transform captures proportional changes
        in spread. Downsampled to N=200 points for speed. Requires minimum 10
        valid observations with spread > 0.

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
        tightening = first - last (within each resampled window)

        Aggregates spread by time window (default 5 minutes) and computes
        mean, std, min, max, first, last, and tightening metrics.

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

        Default percentiles [10, 25, 50, 75, 90] capture the full
        distribution shape for statistical summary.

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
        Regime-conditional spread statistics (mean, std, median, count)
        grouped by microstructure state. See regimes.py for state definitions.

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
