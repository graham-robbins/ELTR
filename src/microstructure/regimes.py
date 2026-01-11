"""
Regime Classification and Markov Transitions.

This module consolidates all regime classification logic including
event-based regimes, microstructure state classification, and
transition matrix computation.

Definitions:
    Event Regimes:
        PREGAME: within pregame_hours of event (default 24h)
        IN_GAME: within ingame_hours after event start (default 3h)
        POST_EVENT: more than ingame_hours after event

        Boundaries are defined by minutes_to_event. Timezone-naive event
        times are localized to UTC. Event time is typically contract
        settlement or underlying event occurrence.

    Microstructure States (Section 3.2, classification hierarchy per Section 3.3):
        FROZEN: v_t = 0 OR v_t < θ_F * v̄_t, θ_F = 0.10 (Eq. 8)
            No meaningful trading activity; price discovery stalled.
        VOLATILITY_BURST: |r_t| > κσ_t AND v_t > λv̄_t, κ = 2.5, λ = 1.5 (Eq. 12)
            Sharp price movement with elevated volume; information shock.
        RESOLUTION_DRIFT: ℓ_t > 0.90 AND spread < 5th pctl AND volume < 25th pctl
                          AND volatility < 25th pctl (Eq. 13)
            Late-stage quiet trading as uncertainty collapses near resolution.
        ACTIVE_INFORMATION: q*_t > θ_A, θ_A = 1.5 (Eq. 11)
            Elevated trading activity consistent with information arrival.
        THIN: s̃_t > θ_T, θ_T = 0.15 (Eq. 9)
            Wide spreads; liquidity scarce, execution costly.
        NORMAL: default state when no other conditions apply (Eq. 10)

    Priority Ordering (Section 3.3, Eq. 14):
        FROZEN > VOLATILITY_BURST > RESOLUTION_DRIFT > ACTIVE_INFORMATION > THIN > NORMAL

    Transition Matrix (Section 3.4, Eq. 16-17):
        P[i,j] = P(S_{t+1} = j | S_t = i)
        Row-stochastic matrix. Self-transitions included. Computed at
        observation-level granularity.

References:
    - Section 3.2: Microstructure state definitions
    - Section 3.3: Priority ordering (Eq. 14)
    - Section 3.4: Markov transition matrix (Eq. 16-17)
    - Section 3.5: Regime entropy (Eq. 18-20)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.types import EventRegime

logger = get_logger("microstructure.regimes")


# State definitions
# Moved from: src/features/feature_engineering.py

class MicrostructureState(Enum):
    """
    Microstructure regime state classification (Section 3.2).

    States are assigned with the following priority (Section 3.3, Eq. 14):
    FROZEN > VOLATILITY_BURST > RESOLUTION_DRIFT > ACTIVE_INFORMATION > THIN > NORMAL

    See module docstring for threshold definitions and equation references.
    """
    FROZEN = auto()
    THIN = auto()
    NORMAL = auto()
    ACTIVE_INFORMATION = auto()
    VOLATILITY_BURST = auto()
    RESOLUTION_DRIFT = auto()
    UNKNOWN = auto()


# Event-based regime classification
# Moved from: src/features/feature_engineering.py (RegimeFeatures class)

def compute_event_regime(
    df: pd.DataFrame,
    event_time: datetime | pd.Timestamp | None = None,
    pregame_hours: int = 24,
    ingame_hours: int = 3,
) -> pd.DataFrame:
    """
    Classify observations into event-based regimes.

    Definition:
        PREGAME: within pregame_hours before event (default 24h)
        IN_GAME: within ingame_hours after event start (default 3h)
        POST_EVENT: more than ingame_hours after event

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    event_time : datetime | pd.Timestamp | None
        Event timestamp.
    pregame_hours : int
        Hours before event for pregame period (default 24).
    ingame_hours : int
        Hours after event start for in-game period (default 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with regime features added.

    Moved from
    ----------
    src/features/feature_engineering.py:444-520 (RegimeFeatures.extract)
    """
    df = df.copy()

    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_market_hours"] = ((df.index.hour >= 9) & (df.index.hour < 16)).astype(int)

    if event_time is not None:
        event_ts = pd.Timestamp(event_time)
        if event_ts.tzinfo is None:
            event_ts = event_ts.tz_localize("UTC")

        df["minutes_to_event"] = (event_ts - df.index).total_seconds() / 60
        df["regime"] = _classify_event_regime(df, pregame_hours, ingame_hours)

    return df


def _classify_event_regime(
    df: pd.DataFrame,
    pregame_hours: int,
    ingame_hours: int,
) -> pd.Series:
    """
    Classify each observation into an event regime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with minutes_to_event column.
    pregame_hours : int
        Hours before event for pregame classification.
    ingame_hours : int
        Hours after event start for in-game classification.

    Returns
    -------
    pd.Series
        Regime classification for each observation.

    Moved from
    ----------
    src/features/feature_engineering.py:493-510 (RegimeFeatures._classify_regime)
    """
    regimes = pd.Series(EventRegime.UNKNOWN.value, index=df.index)

    if "minutes_to_event" not in df.columns:
        return regimes

    mtoe = df["minutes_to_event"]
    ingame_minutes = ingame_hours * 60

    pregame_mask = (mtoe > 0) & (mtoe <= pregame_hours * 60)
    ingame_mask = (mtoe <= 0) & (mtoe > -ingame_minutes)
    post_mask = mtoe <= -ingame_minutes

    regimes[pregame_mask] = EventRegime.PREGAME.value
    regimes[ingame_mask] = EventRegime.IN_GAME.value
    regimes[post_mask] = EventRegime.POST_EVENT.value

    return regimes


# Microstructure state classification
# Moved from: src/features/feature_engineering.py (MicrostructureRegimeFeatures)

def compute_microstructure_regime(
    df: pd.DataFrame,
    frozen_volume_threshold: float = 0.1,
    thin_spread_threshold: float = 0.15,
    active_volume_zscore: float = 1.5,
    burst_volatility_k: float = 2.5,
    burst_volume_multiplier: float = 1.5,
    resolution_lifecycle_threshold: float = 0.90,
    resolution_spread_threshold: float = 0.05,
    resolution_volume_quantile: float = 0.25,
    resolution_volatility_quantile: float = 0.25,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """
    Classify microstructure state for each observation (Section 3.2).

    Definition:
        Rule-based classification using RAW (non-normalized) values.
        Priority ordering per Section 3.3, Eq. 14:
            FROZEN > VOLATILITY_BURST > RESOLUTION_DRIFT > ACTIVE_INFORMATION > THIN > NORMAL

        FROZEN is evaluated last but has highest priority (overwrites all).
        See module docstring for threshold definitions and equation references.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume, spread, and volatility features.
    frozen_volume_threshold : float
        Volume threshold for FROZEN state (default 0.1).
    thin_spread_threshold : float
        Spread percentage threshold for THIN state (default 0.15).
    active_volume_zscore : float
        Volume z-score threshold for ACTIVE_INFORMATION (default 1.5).
    burst_volatility_k : float
        Volatility multiplier for VOLATILITY_BURST (default 2.5).
    burst_volume_multiplier : float
        Volume multiplier for VOLATILITY_BURST (default 1.5).
    resolution_lifecycle_threshold : float
        Lifecycle ratio threshold for RESOLUTION_DRIFT (default 0.90).
    resolution_spread_threshold : float
        Spread threshold for RESOLUTION_DRIFT (default 0.05).
    resolution_volume_quantile : float
        Volume quantile for RESOLUTION_DRIFT (default 0.25).
    resolution_volatility_quantile : float
        Volatility quantile for RESOLUTION_DRIFT (default 0.25).
    rolling_window : int
        Rolling window for baseline computation (default 20).

    Returns
    -------
    pd.DataFrame
        DataFrame with microstructure_state column.

    Moved from
    ----------
    src/features/feature_engineering.py:629-743 (MicrostructureRegimeFeatures.extract)
    """
    df = df.copy()
    states = pd.Series(MicrostructureState.NORMAL.value, index=df.index)

    # Compute raw rolling metrics
    raw_rolling_volatility = None
    raw_rolling_volume = None

    if "pct_return" in df.columns:
        raw_rolling_volatility = df["pct_return"].abs().rolling(
            window=rolling_window, min_periods=3
        ).mean()

    if "volume" in df.columns:
        raw_rolling_volume = df["volume"].rolling(
            window=rolling_window, min_periods=1
        ).mean()

    # Volatility burst detection (Section 3.2, Eq. 12)
    if raw_rolling_volatility is not None and "pct_return" in df.columns:
        mid_return = df["pct_return"].abs()
        volatility_condition = mid_return > (burst_volatility_k * raw_rolling_volatility)

        if raw_rolling_volume is not None:
            volume_condition = df["volume"] > (raw_rolling_volume * burst_volume_multiplier)
            burst_mask = volatility_condition & volume_condition
        else:
            burst_mask = volatility_condition

        states[burst_mask] = MicrostructureState.VOLATILITY_BURST.value

    # Active information arrival (Section 3.2, Eq. 11)
    if raw_rolling_volume is not None and "volume" in df.columns:
        vol_std = df["volume"].rolling(window=rolling_window, min_periods=1).std()
        vol_zscore = np.where(
            vol_std > 0,
            (df["volume"] - raw_rolling_volume) / vol_std,
            0,
        )
        active_volume_mask = vol_zscore > active_volume_zscore
        active_info_mask = active_volume_mask & (states == MicrostructureState.NORMAL.value)
        states[active_info_mask] = MicrostructureState.ACTIVE_INFORMATION.value

    # Resolution drift (Section 3.2, Eq. 13)
    if "lifecycle_ratio" in df.columns:
        lifecycle_mask = df["lifecycle_ratio"] > resolution_lifecycle_threshold

        spread_ok = pd.Series(True, index=df.index)
        volume_ok = pd.Series(True, index=df.index)
        volatility_ok = pd.Series(True, index=df.index)

        if "spread_pct" in df.columns:
            spread_threshold = df["spread_pct"].quantile(resolution_spread_threshold)
            spread_ok = df["spread_pct"] < max(spread_threshold, 0.05)

        if "volume" in df.columns:
            volume_threshold = df["volume"].quantile(resolution_volume_quantile)
            volume_ok = df["volume"] < volume_threshold

        if raw_rolling_volatility is not None:
            vol_threshold = raw_rolling_volatility.quantile(resolution_volatility_quantile)
            volatility_ok = raw_rolling_volatility < vol_threshold

        resolution_mask = lifecycle_mask & spread_ok & volume_ok & volatility_ok
        # Priority: only assign if not already VOLATILITY_BURST (Section 3.3)
        resolution_mask = resolution_mask & (states != MicrostructureState.VOLATILITY_BURST.value)
        states[resolution_mask] = MicrostructureState.RESOLUTION_DRIFT.value

    # Thin market (Section 3.2, Eq. 9)
    if "spread_pct" in df.columns:
        thin_mask = df["spread_pct"] > thin_spread_threshold
        thin_mask = thin_mask & (states == MicrostructureState.NORMAL.value)
        states[thin_mask] = MicrostructureState.THIN.value

    # Frozen market (Section 3.2, Eq. 8)
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(window=rolling_window, min_periods=1).mean()
        frozen_mask = df["volume"] < (vol_ma * frozen_volume_threshold)
        frozen_mask = frozen_mask | (df["volume"] == 0)
        states[frozen_mask] = MicrostructureState.FROZEN.value

    df["microstructure_state"] = states
    df["microstructure_state_name"] = df["microstructure_state"].map({
        MicrostructureState.FROZEN.value: "frozen",
        MicrostructureState.THIN.value: "thin",
        MicrostructureState.NORMAL.value: "normal",
        MicrostructureState.ACTIVE_INFORMATION.value: "active_info",
        MicrostructureState.VOLATILITY_BURST.value: "volatility_burst",
        MicrostructureState.RESOLUTION_DRIFT.value: "resolution_drift",
        MicrostructureState.UNKNOWN.value: "unknown",
    })
    df["regime_transition"] = (
        df["microstructure_state"] != df["microstructure_state"].shift(1)
    ).astype(int)

    return df


# Markov transition matrix
# Moved from: src/utils/export.py

def compute_transition_matrix(
    dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute regime transition probability matrix.

    Definition:
        P[i,j] = count(state_i -> state_j) / sum(transitions from state_i)
        Row-stochastic matrix where rows sum to 1.

        Self-transitions (i=j) are included. Transitions are counted at
        observation-level granularity (consecutive state pairs). Assumes
        approximate time-homogeneity within the analysis period.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of DataFrames with microstructure_state column.

    Returns
    -------
    pd.DataFrame
        Transition matrix with states as rows and columns.

    Moved from
    ----------
    src/utils/export.py:528-601 (MetricsExporter.export_regime_transitions)
    """
    all_regimes = set()
    transition_counts = {}

    for df in dfs:
        if "microstructure_state" not in df.columns:
            continue

        states = df["microstructure_state"].astype(str).str.replace(
            "MicrostructureState.", "", regex=False
        )
        all_regimes.update(states.unique())

        for i in range(len(states) - 1):
            from_state = states.iloc[i]
            to_state = states.iloc[i + 1]
            key = (from_state, to_state)
            transition_counts[key] = transition_counts.get(key, 0) + 1

    if not all_regimes:
        return pd.DataFrame()

    all_regimes = sorted(all_regimes)
    n_regimes = len(all_regimes)

    matrix = np.zeros((n_regimes, n_regimes))
    for (from_state, to_state), count in transition_counts.items():
        if from_state in all_regimes and to_state in all_regimes:
            i = all_regimes.index(from_state)
            j = all_regimes.index(to_state)
            matrix[i, j] = count

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        matrix_norm = np.divide(matrix, row_sums, where=row_sums > 0)
        matrix_norm = np.nan_to_num(matrix_norm)

    return pd.DataFrame(matrix_norm, index=all_regimes, columns=all_regimes)


def compute_regime_entropy(df: pd.DataFrame) -> float:
    """
    Compute normalized regime entropy.

    Definition:
        Entropy = -sum(p_i * log2(p_i)) / log2(num_states)
        Normalized to [0, 1] where 1 = uniform distribution

        Normalization enables comparison across contracts with different
        numbers of observed states. Higher entropy indicates more diverse
        regime distribution; lower entropy indicates dominance by few states.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with microstructure_state column.

    Returns
    -------
    float
        Normalized entropy (0 to 1).

    Moved from
    ----------
    src/utils/export.py:231-275 (MetricsExporter._compute_regime_metrics)
    """
    if "microstructure_state" not in df.columns:
        return np.nan

    state_counts = df["microstructure_state"].value_counts()
    total = state_counts.sum()

    if total == 0:
        return np.nan

    probs = (state_counts / total).values
    probs = probs[probs > 0]
    num_states = len(probs)

    if num_states <= 1:
        return 0.0

    raw_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_states)

    return raw_entropy / max_entropy if max_entropy > 0 else 0.0
