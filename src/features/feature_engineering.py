"""
Feature engineering module for IRP platform.

Provides comprehensive feature extraction for prediction market
timeseries including returns, volatility, liquidity, spread metrics,
regime classification, and contract lifecycle normalization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum, auto
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    ContractTimeseries,
    DataFrequency,
    EventRegime,
    MarketDataset,
)
from src.utils.time_binning import (
    compute_lifecycle_features as _compute_lifecycle_features,
    resample_to_1min,
)

logger = get_logger("features")


class MicrostructureState(Enum):
    """Microstructure regime state classification."""
    FROZEN = auto()
    THIN = auto()
    NORMAL = auto()
    ACTIVE_INFORMATION = auto()
    VOLATILITY_BURST = auto()
    RESOLUTION_DRIFT = auto()
    UNKNOWN = auto()


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Feature extractor name."""
        pass

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features and return augmented DataFrame."""
        pass


class ReturnFeatures(FeatureExtractor):
    """
    Extracts return-based features.

    Computes percent returns and logit returns from price data.
    """

    @property
    def name(self) -> str:
        return "returns"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract return features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with return features added.
        """
        df = df.copy()

        price_col = self._get_price_column(df)
        if price_col is None:
            return df

        df["pct_return"] = df[price_col].pct_change()
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["logit_return"] = self._compute_logit_return(df[price_col])
        df["abs_return"] = df["pct_return"].abs()

        return df

    def _get_price_column(self, df: pd.DataFrame) -> str | None:
        """Get primary price column."""
        candidates = ["price_c", "price", "close", "last_price"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _compute_logit_return(self, prices: pd.Series) -> pd.Series:
        """
        Compute logit returns.

        Logit transform: log(p / (1-p))
        Logit return: logit(p_t) - logit(p_{t-1})
        """
        p = prices / 100  # Convert cents to probability
        p = p.clip(0.001, 0.999)  # Avoid log(0) issues
        logit = np.log(p / (1 - p))
        return logit.diff()


class VolatilityFeatures(FeatureExtractor):
    """
    Extracts volatility-based features.

    Computes rolling volatility measures at multiple window sizes.
    """

    def __init__(
        self,
        windows: dict[str, int] | None = None,
        min_periods: int = 3,
        annualization_factor: int = 525600,
    ):
        self.windows = windows or {"short": 5, "medium": 15, "long": 60}
        self.min_periods = min_periods
        self.annualization_factor = annualization_factor

    @property
    def name(self) -> str:
        return "volatility"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volatility features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with return columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with volatility features added.
        """
        df = df.copy()

        return_col = self._get_return_column(df)
        if return_col is None:
            return df

        for name, window in self.windows.items():
            col_name = f"volatility_{name}"
            df[col_name] = (
                df[return_col]
                .rolling(window=window, min_periods=self.min_periods)
                .std()
            )

            df[f"{col_name}_ann"] = df[col_name] * np.sqrt(self.annualization_factor)

        df["realized_variance"] = df[return_col] ** 2

        high_col = self._find_column(df, ["price_h", "high"])
        low_col = self._find_column(df, ["price_l", "low"])
        if high_col and low_col:
            df["parkinson_vol"] = self._parkinson_volatility(df[high_col], df[low_col])
            df["price_range"] = df[high_col] - df[low_col]

        return df

    def _get_return_column(self, df: pd.DataFrame) -> str | None:
        """Get primary return column."""
        candidates = ["pct_return", "log_return", "return"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find first matching column."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _parkinson_volatility(
        self, high: pd.Series, low: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Compute Parkinson volatility estimator.

        More efficient than close-to-close volatility.

        Handles edge cases where low=0 or high/low ratio is invalid.
        """
        # Guard against division by zero: replace 0 with NaN
        safe_low = low.replace(0, np.nan)

        # Compute ratio only where valid
        ratio = high / safe_low

        # Additional guard: ratio must be positive for log
        ratio = ratio.where(ratio > 0, np.nan)

        log_hl = np.log(ratio) ** 2
        factor = 1 / (4 * np.log(2))
        return np.sqrt(factor * log_hl.rolling(window=window).mean())


class LiquidityFeatures(FeatureExtractor):
    """
    Extracts liquidity-based features.

    Computes volume metrics, turnover, and depth indicators.
    """

    def __init__(self, volume_ma_window: int = 20):
        self.volume_ma_window = volume_ma_window

    @property
    def name(self) -> str:
        return "liquidity"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract liquidity features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with volume data.

        Returns
        -------
        pd.DataFrame
            DataFrame with liquidity features added.
        """
        df = df.copy()

        if "volume" not in df.columns:
            return df

        df["volume_ma"] = df["volume"].rolling(
            window=self.volume_ma_window, min_periods=1
        ).mean()

        df["volume_std"] = df["volume"].rolling(
            window=self.volume_ma_window, min_periods=1
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


class SpreadFeatures(FeatureExtractor):
    """
    Extracts bid-ask spread features.

    Computes spread metrics, midpoint prices, and spread dynamics.
    """

    def __init__(self, basis_points: bool = True):
        self.basis_points = basis_points

    @property
    def name(self) -> str:
        return "spread"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spread features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with bid/ask data.

        Returns
        -------
        pd.DataFrame
            DataFrame with spread features added.
        """
        df = df.copy()

        bid_col = self._find_column(df, ["yes_bid_c", "bid", "bid_price"])
        ask_col = self._find_column(df, ["yes_ask_c", "ask", "ask_price"])

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

        if self.basis_points:
            df["spread_bps"] = df["spread_pct"] * 10000

        df["spread_ma"] = df["spread"].rolling(window=20, min_periods=1).mean()

        df["spread_tightening"] = df["spread"].diff()

        df["effective_spread"] = self._compute_effective_spread(df, bid_col, ask_col)

        return df

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find first matching column."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _compute_effective_spread(
        self, df: pd.DataFrame, bid_col: str, ask_col: str
    ) -> pd.Series:
        """
        Compute effective spread using trade prices.

        Effective spread = 2 * |trade_price - midpoint|
        """
        price_col = self._find_column(df, ["price_c", "price", "close"])
        if price_col is None:
            return pd.Series(np.nan, index=df.index)

        midpoint = (df[bid_col] + df[ask_col]) / 2
        return 2 * (df[price_col] - midpoint).abs()


class DepthFeatures(FeatureExtractor):
    """
    Extracts order book depth features.

    Analyzes depth resilience and book thinning metrics.
    """

    @property
    def name(self) -> str:
        return "depth"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract depth features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC bid/ask data.

        Returns
        -------
        pd.DataFrame
            DataFrame with depth features added.
        """
        df = df.copy()

        bid_h = self._find_column(df, ["yes_bid_h", "bid_high"])
        bid_l = self._find_column(df, ["yes_bid_l", "bid_low"])
        ask_h = self._find_column(df, ["yes_ask_h", "ask_high"])
        ask_l = self._find_column(df, ["yes_ask_l", "ask_low"])

        if bid_h and bid_l:
            df["bid_range"] = df[bid_h] - df[bid_l]

        if ask_h and ask_l:
            df["ask_range"] = df[ask_h] - df[ask_l]

        if bid_h and bid_l and ask_h and ask_l:
            df["book_thinning"] = (df["bid_range"] + df["ask_range"]) / 2

        if "spread" in df.columns:
            df["depth_resilience"] = self._compute_depth_resilience(df)

        return df

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find first matching column."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _compute_depth_resilience(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Compute depth resilience metric.

        Measures how quickly spreads return to normal after widening.
        """
        spread_change = df["spread"].diff()
        spread_recovery = spread_change.rolling(window=window).apply(
            lambda x: (x < 0).sum() / len(x) if len(x) > 0 else np.nan,
            raw=False,
        )
        return spread_recovery


class RegimeFeatures(FeatureExtractor):
    """
    Extracts regime classification features.

    Identifies market regimes: pregame, in-game, halftime,
    event-release, post-event.
    """

    def __init__(
        self,
        pregame_hours: int = 24,
        halftime_buffer_minutes: int = 15,
        event_time: datetime | None = None,
    ):
        self.pregame_hours = pregame_hours
        self.halftime_buffer_minutes = halftime_buffer_minutes
        self.event_time = event_time

    @property
    def name(self) -> str:
        return "regime"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract regime features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with datetime index.

        Returns
        -------
        pd.DataFrame
            DataFrame with regime features added.
        """
        df = df.copy()

        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["is_market_hours"] = self._is_market_hours(df.index)

        if self.event_time is not None:
            df["minutes_to_event"] = self._compute_minutes_to_event(
                df.index, self.event_time
            )
            df["regime"] = self._classify_regime(df)

        df["session_start"] = self._is_session_start(df.index)
        df["session_end"] = self._is_session_end(df.index)

        return df

    def _is_market_hours(self, index: pd.DatetimeIndex) -> pd.Series:
        """Check if timestamps are during US market hours."""
        hours = index.hour
        return ((hours >= 9) & (hours < 16)).astype(int)

    def _compute_minutes_to_event(
        self, index: pd.DatetimeIndex, event_time: datetime
    ) -> pd.Series:
        """Compute minutes until event."""
        event_ts = pd.Timestamp(event_time)
        if event_ts.tzinfo is None:
            event_ts = event_ts.tz_localize("UTC")

        deltas = (event_ts - index).total_seconds() / 60
        return pd.Series(deltas, index=index)

    def _classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify each observation into a regime."""
        regimes = pd.Series(EventRegime.UNKNOWN.value, index=df.index)

        if "minutes_to_event" not in df.columns:
            return regimes

        mtoe = df["minutes_to_event"]

        pregame_mask = (mtoe > 0) & (mtoe <= self.pregame_hours * 60)
        ingame_mask = (mtoe <= 0) & (mtoe > -180)  # First 3 hours after start
        post_mask = mtoe <= -180

        regimes[pregame_mask] = EventRegime.PREGAME.value
        regimes[ingame_mask] = EventRegime.IN_GAME.value
        regimes[post_mask] = EventRegime.POST_EVENT.value

        return regimes

    def _is_session_start(self, index: pd.DatetimeIndex) -> pd.Series:
        """Identify session start times."""
        hour_minute = index.hour * 60 + index.minute
        return (hour_minute >= 9 * 60) & (hour_minute < 9 * 60 + 5)

    def _is_session_end(self, index: pd.DatetimeIndex) -> pd.Series:
        """Identify session end times."""
        hour_minute = index.hour * 60 + index.minute
        return (hour_minute >= 16 * 60 - 5) & (hour_minute < 16 * 60)


class LifecycleFeatures(FeatureExtractor):
    """
    Extracts contract lifecycle normalization features.

    Delegates to compute_lifecycle_features() in time_binning.py
    which is the SINGLE SOURCE OF TRUTH for lifecycle computation.
    """

    def __init__(
        self,
        listing_time: datetime | None = None,
        settlement_time: datetime | None = None,
        event_time: datetime | None = None,
    ):
        self.listing_time = listing_time
        self.settlement_time = settlement_time
        self.event_time = event_time

    @property
    def name(self) -> str:
        return "lifecycle"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lifecycle normalization features.

        Delegates to time_binning.compute_lifecycle_features() for
        consistent lifecycle computation across the platform.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame with lifecycle features:
            - tsl_hours: Time since listing in hours
            - tts_hours: Time to settlement in hours
            - lifecycle_ratio: Normalized 0→1 lifecycle position
            - lifecycle_phase: Categorical phase using nonlinear bins
        """
        listing = self.listing_time
        settlement = self.settlement_time
        event = self.event_time

        if isinstance(listing, datetime):
            listing = pd.Timestamp(listing)
        if isinstance(settlement, datetime):
            settlement = pd.Timestamp(settlement)
        if isinstance(event, datetime):
            event = pd.Timestamp(event)

        return _compute_lifecycle_features(
            df,
            listing_time=listing,
            settlement_time=settlement,
            event_time=event,
        )


class MicrostructureRegimeFeatures(FeatureExtractor):
    """
    Classifies microstructure state at each observation (Section 3.2).

    Uses RAW volatility and volume values (not normalized).
    Priority ordering per Section 3.3, Eq. 14:
        FROZEN > VOLATILITY_BURST > RESOLUTION_DRIFT > ACTIVE_INFORMATION > THIN > NORMAL

    State definitions (Section 3.2):
    - Frozen: v_t = 0 OR v_t < θ_F * v̄_t, θ_F = 0.10 (Eq. 8)
    - Thin: s̃_t > θ_T, θ_T = 0.15 (Eq. 9)
    - Normal: default state when no other conditions apply (Eq. 10)
    - Active Information: q*_t > θ_A, θ_A = 1.5 (Eq. 11)
    - Volatility Burst: |r_t| > κσ_t AND v_t > λv̄_t, κ = 2.5, λ = 1.5 (Eq. 12)
    - Resolution Drift: ℓ_t > 0.90 AND low spread/volume/volatility (Eq. 13)
    """

    def __init__(
        self,
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
    ):
        self.frozen_volume_threshold = frozen_volume_threshold
        self.thin_spread_threshold = thin_spread_threshold
        self.active_volume_zscore = active_volume_zscore
        self.burst_volatility_k = burst_volatility_k
        self.burst_volume_multiplier = burst_volume_multiplier
        self.resolution_lifecycle_threshold = resolution_lifecycle_threshold
        self.resolution_spread_threshold = resolution_spread_threshold
        self.resolution_volume_quantile = resolution_volume_quantile
        self.resolution_volatility_quantile = resolution_volatility_quantile
        self.rolling_window = rolling_window

    @property
    def name(self) -> str:
        return "microstructure_regime"

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify microstructure state for each observation.

        Uses RAW volatility and volume values (not normalized).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with volume, spread, and volatility features.

        Returns
        -------
        pd.DataFrame
            DataFrame with microstructure_state column.
        """
        df = df.copy()

        states = pd.Series(MicrostructureState.NORMAL.value, index=df.index)

        # Compute raw rolling metrics for regime classification
        raw_rolling_volatility = None
        raw_rolling_volume = None

        if "pct_return" in df.columns:
            raw_rolling_volatility = df["pct_return"].abs().rolling(
                window=self.rolling_window, min_periods=3
            ).mean()

        if "volume" in df.columns:
            raw_rolling_volume = df["volume"].rolling(
                window=self.rolling_window, min_periods=1
            ).mean()

        # Volatility burst detection (Section 3.2, Eq. 12):
        # |r_t| > κσ_t AND v_t > λv̄_t, κ = 2.5, λ = 1.5
        if raw_rolling_volatility is not None and "pct_return" in df.columns:
            mid_return = df["pct_return"].abs()
            volatility_condition = mid_return > (self.burst_volatility_k * raw_rolling_volatility)

            if raw_rolling_volume is not None:
                volume_condition = df["volume"] > (raw_rolling_volume * self.burst_volume_multiplier)
                burst_mask = volatility_condition & volume_condition
            else:
                burst_mask = volatility_condition

            states[burst_mask] = MicrostructureState.VOLATILITY_BURST.value

        # Active information arrival (Section 3.2, Eq. 11): q*_t > θ_A, θ_A = 1.5
        if raw_rolling_volume is not None and "volume" in df.columns:
            vol_std = df["volume"].rolling(window=self.rolling_window, min_periods=1).std()
            vol_zscore = np.where(
                vol_std > 0,
                (df["volume"] - raw_rolling_volume) / vol_std,
                0,
            )
            active_volume_mask = vol_zscore > self.active_volume_zscore
            # Only mark as active info if not already a volatility burst
            active_info_mask = active_volume_mask & (states == MicrostructureState.NORMAL.value)
            states[active_info_mask] = MicrostructureState.ACTIVE_INFORMATION.value

        # Resolution drift (Section 3.2, Eq. 13):
        # ℓ_t > 0.90 AND spread < 5th pctl AND volume < 25th pctl AND volatility < 25th pctl
        if "lifecycle_ratio" in df.columns:
            lifecycle_mask = df["lifecycle_ratio"] > self.resolution_lifecycle_threshold

            # Compute thresholds from contract-level quantiles
            spread_ok = pd.Series(True, index=df.index)
            volume_ok = pd.Series(True, index=df.index)
            volatility_ok = pd.Series(True, index=df.index)

            if "spread_pct" in df.columns:
                spread_threshold = df["spread_pct"].quantile(self.resolution_spread_threshold)
                spread_ok = df["spread_pct"] < max(spread_threshold, 0.05)

            if "volume" in df.columns:
                volume_threshold = df["volume"].quantile(self.resolution_volume_quantile)
                volume_ok = df["volume"] < volume_threshold

            if raw_rolling_volatility is not None:
                vol_threshold = raw_rolling_volatility.quantile(self.resolution_volatility_quantile)
                volatility_ok = raw_rolling_volatility < vol_threshold

            resolution_mask = lifecycle_mask & spread_ok & volume_ok & volatility_ok
            # Priority: only assign if not already VOLATILITY_BURST (Section 3.3)
            resolution_mask = resolution_mask & (states != MicrostructureState.VOLATILITY_BURST.value)
            states[resolution_mask] = MicrostructureState.RESOLUTION_DRIFT.value

        # Thin market (Section 3.2, Eq. 9): s̃_t > θ_T, θ_T = 0.15
        if "spread_pct" in df.columns:
            thin_mask = df["spread_pct"] > self.thin_spread_threshold
            thin_mask = thin_mask & (states == MicrostructureState.NORMAL.value)
            states[thin_mask] = MicrostructureState.THIN.value

        # Frozen market (Section 3.2, Eq. 8): v_t = 0 OR v_t < θ_F * v̄_t, θ_F = 0.10
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(window=self.rolling_window, min_periods=1).mean()
            frozen_mask = df["volume"] < (vol_ma * self.frozen_volume_threshold)
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

        df["regime_transition"] = (df["microstructure_state"] != df["microstructure_state"].shift(1)).astype(int)

        return df


class FeatureEngineer:
    """
    Main feature engineering orchestrator.

    Coordinates all feature extractors and produces
    comprehensive feature sets.
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize feature engineer.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration. Uses global if None.
        """
        self.config = config or get_config()
        self.feature_config = self.config.features

        self.extractors: list[FeatureExtractor] = [
            ReturnFeatures(),
            VolatilityFeatures(
                windows={
                    "short": self.feature_config.rolling_windows.short,
                    "medium": self.feature_config.rolling_windows.medium,
                    "long": self.feature_config.rolling_windows.long,
                },
                min_periods=self.feature_config.volatility.min_periods,
                annualization_factor=self.feature_config.volatility.annualization_factor,
            ),
            LiquidityFeatures(
                volume_ma_window=self.feature_config.liquidity.volume_ma_window,
            ),
            SpreadFeatures(
                basis_points=self.feature_config.spread.basis_points,
            ),
            DepthFeatures(),
            RegimeFeatures(
                pregame_hours=self.config.microstructure.regime_detection.pregame_hours,
                halftime_buffer_minutes=self.config.microstructure.regime_detection.halftime_buffer_minutes,
            ),
            LifecycleFeatures(),
            MicrostructureRegimeFeatures(
                frozen_volume_threshold=self.config.microstructure.regime_classification.frozen_volume_threshold,
                thin_spread_threshold=self.config.microstructure.regime_classification.thin_spread_threshold,
                active_volume_zscore=self.config.microstructure.regime_classification.active_volume_zscore,
                burst_volatility_k=getattr(
                    self.config.microstructure.regime_classification, "burst_volatility_k", 2.5
                ),
                burst_volume_multiplier=getattr(
                    self.config.microstructure.regime_classification, "burst_volume_multiplier", 1.5
                ),
                resolution_lifecycle_threshold=self.config.microstructure.regime_classification.resolution_lifecycle_threshold,
                resolution_spread_threshold=getattr(
                    self.config.microstructure.regime_classification, "resolution_spread_threshold", 0.05
                ),
                resolution_volume_quantile=getattr(
                    self.config.microstructure.regime_classification, "resolution_volume_quantile", 0.25
                ),
                resolution_volatility_quantile=getattr(
                    self.config.microstructure.regime_classification, "resolution_volatility_quantile", 0.25
                ),
            ),
        ]

    def engineer_features(self, contract: ContractTimeseries) -> ContractTimeseries:
        """
        Engineer features for single contract.

        Resamples to 1-minute intervals before computing rolling metrics
        to prevent false volatility bursts from irregular sampling (Section 3).

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to process.

        Returns
        -------
        ContractTimeseries
            Contract with engineered features.
        """
        df = contract.data.copy()

        # Resample to 1-minute intervals before computing rolling metrics (Section 3)
        # This prevents false volatility bursts from irregular sampling
        try:
            df = resample_to_1min(df)
        except Exception as e:
            logger.warning(
                f"Resampling failed for {contract.contract_id}: {e}. "
                "Proceeding with original data."
            )

        for extractor in self.extractors:
            try:
                df = extractor.extract(df)
            except Exception as e:
                logger.warning(
                    f"Feature extraction failed for {contract.contract_id} "
                    f"({extractor.name}): {e}"
                )

        return ContractTimeseries(
            contract_id=contract.contract_id,
            category=contract.category,
            data=df,
            metadata=contract.metadata,
            frequency=contract.frequency,
            event_time=contract.event_time,
        )

    def engineer_dataset(self, dataset: MarketDataset) -> MarketDataset:
        """
        Engineer features for entire dataset.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to process.

        Returns
        -------
        MarketDataset
            Dataset with engineered features.
        """
        logger.info(f"Engineering features for {len(dataset)} contracts")

        featured_dataset = MarketDataset(metadata=dataset.metadata)

        for contract in dataset:
            featured_contract = self.engineer_features(contract)
            featured_dataset.add(featured_contract)

        logger.info("Feature engineering complete")
        return featured_dataset

    def add_extractor(self, extractor: FeatureExtractor) -> None:
        """Add custom feature extractor."""
        self.extractors.append(extractor)

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names produced."""
        dummy_df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
            "price_o": [49.0, 50.0, 51.0],
            "price_h": [51.0, 52.0, 53.0],
            "price_l": [48.0, 49.0, 50.0],
            "volume": [100, 150, 200],
            "trade_count": [10, 15, 20],
            "yes_bid_c": [49.0, 50.0, 51.0],
            "yes_ask_c": [51.0, 52.0, 53.0],
            "yes_bid_h": [50.0, 51.0, 52.0],
            "yes_bid_l": [48.0, 49.0, 50.0],
            "yes_ask_h": [52.0, 53.0, 54.0],
            "yes_ask_l": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min"))

        original_cols = set(dummy_df.columns)

        for extractor in self.extractors:
            dummy_df = extractor.extract(dummy_df)

        new_cols = set(dummy_df.columns) - original_cols
        return sorted(list(new_cols))


def engineer_features(
    dataset: MarketDataset, config: IRPConfig | None = None
) -> MarketDataset:
    """
    Convenience function to engineer features for dataset.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to process.
    config : IRPConfig | None
        Platform configuration.

    Returns
    -------
    MarketDataset
        Dataset with engineered features.
    """
    engineer = FeatureEngineer(config)
    return engineer.engineer_dataset(dataset)
