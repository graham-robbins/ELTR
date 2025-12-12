"""
Microstructure analysis module for IRP platform.

Provides advanced market microstructure analytics including
liquidity surges, volatility dynamics, spread evolution,
and event-aligned trajectory analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    AnalyticsResult,
    Category,
    ContractID,
    ContractTimeseries,
    EventRegime,
    MarketDataset,
)

logger = get_logger("microstructure")


@dataclass
class SurgeEvent:
    """Container for detected surge events."""
    contract_id: ContractID
    timestamp: pd.Timestamp
    surge_type: str
    magnitude: float
    baseline: float
    duration_minutes: int | None = None


@dataclass
class EventWindow:
    """Container for event-aligned analysis window."""
    contract_id: ContractID
    event_time: pd.Timestamp
    pre_event_data: pd.DataFrame
    post_event_data: pd.DataFrame
    event_type: str | None = None


@dataclass
class MicrostructureMetrics:
    """Container for microstructure summary metrics."""
    contract_id: ContractID
    category: Category
    avg_spread: float
    avg_spread_pct: float
    avg_volume: float
    volatility: float
    liquidity_score: float
    depth_resilience: float
    surge_count: int
    metrics: dict[str, float] = field(default_factory=dict)


class SurgeDetector:
    """
    Detects liquidity and volatility surges.

    Identifies abnormal market activity based on
    rolling statistical thresholds.
    """

    def __init__(
        self,
        volume_threshold: float = 2.0,
        volatility_threshold: float = 2.0,
        lookback_window: int = 30,
    ):
        self.volume_threshold = volume_threshold
        self.volatility_threshold = volatility_threshold
        self.lookback_window = lookback_window

    def detect_volume_surges(
        self, df: pd.DataFrame, contract_id: ContractID
    ) -> list[SurgeEvent]:
        """
        Detect volume surge events.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with volume data.
        contract_id : ContractID
            Contract identifier.

        Returns
        -------
        list[SurgeEvent]
            Detected surge events.
        """
        if "volume" not in df.columns:
            return []

        volume = df["volume"]
        rolling_mean = volume.rolling(window=self.lookback_window, min_periods=5).mean()
        rolling_std = volume.rolling(window=self.lookback_window, min_periods=5).std()

        zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)

        surge_mask = zscore > self.volume_threshold
        surges = []

        for idx in df.index[surge_mask]:
            surges.append(SurgeEvent(
                contract_id=contract_id,
                timestamp=idx,
                surge_type="volume",
                magnitude=float(zscore.loc[idx]),
                baseline=float(rolling_mean.loc[idx]),
            ))

        return surges

    def detect_volatility_surges(
        self, df: pd.DataFrame, contract_id: ContractID
    ) -> list[SurgeEvent]:
        """
        Detect volatility surge events.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with return data.
        contract_id : ContractID
            Contract identifier.

        Returns
        -------
        list[SurgeEvent]
            Detected surge events.
        """
        return_col = None
        for col in ["abs_return", "pct_return", "log_return"]:
            if col in df.columns:
                return_col = col
                break

        if return_col is None:
            return []

        returns = df[return_col].abs() if return_col != "abs_return" else df[return_col]
        rolling_mean = returns.rolling(window=self.lookback_window, min_periods=5).mean()
        rolling_std = returns.rolling(window=self.lookback_window, min_periods=5).std()

        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        surge_mask = zscore > self.volatility_threshold
        surges = []

        for idx in df.index[surge_mask]:
            if pd.isna(zscore.loc[idx]):
                continue
            surges.append(SurgeEvent(
                contract_id=contract_id,
                timestamp=idx,
                surge_type="volatility",
                magnitude=float(zscore.loc[idx]),
                baseline=float(rolling_mean.loc[idx]),
            ))

        return surges

    def compute_surge_ratio(self, df: pd.DataFrame, metric: str = "volume") -> pd.Series:
        """
        Compute rolling surge ratio.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        metric : str
            Column to compute ratio for.

        Returns
        -------
        pd.Series
            Surge ratio time series.
        """
        if metric not in df.columns:
            return pd.Series(np.nan, index=df.index)

        values = df[metric]
        rolling_mean = values.rolling(window=self.lookback_window, min_periods=5).mean()
        return values / rolling_mean.replace(0, np.nan)


class SpreadAnalyzer:
    """
    Analyzes bid-ask spread dynamics.

    Computes spread tightening curves, spread evolution metrics,
    robust spread collapse slope (Theil-Sen), and liquidity resilience.
    """

    def robust_spread_collapse_slope(
        self, df: pd.DataFrame, lifecycle_col: str = "lifecycle_ratio"
    ) -> dict[str, float]:
        """
        Fast slope estimate of log(spread) over lifecycle.

        Replaces theilslopes with OLS + downsampling for performance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with spread and lifecycle_ratio columns.
        lifecycle_col : str
            Lifecycle column name.

        Returns
        -------
        dict[str, float]
            Slope estimate (OLS-based for speed).
        """
        if "spread" not in df.columns or lifecycle_col not in df.columns:
            return {"spread_collapse_slope": np.nan}

        # Extract arrays
        spread = df["spread"].to_numpy()
        lifecycle = df[lifecycle_col].to_numpy()

        # Clean
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

        # Transform
        log_spread = np.log(spread)

        # Fast OLS slope
        den = np.var(lifecycle)
        if den == 0:
            return {"spread_collapse_slope": np.nan}

        num = np.cov(lifecycle, log_spread, bias=True)[0, 1]
        return {"spread_collapse_slope": num / den}

    def compute_liquidity_resilience(
        self,
        df: pd.DataFrame,
        shock_threshold_quantile: float = 0.95,
        recovery_window_minutes: int = 30,
    ) -> dict[str, float]:
        """
        Compute liquidity resilience via spread-shock recovery time (Section 6).

        Identifies spread shocks (spread > 95th percentile) and measures
        the time to return to median spread.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with spread and DatetimeIndex.
        shock_threshold_quantile : float
            Quantile threshold for shock detection.
        recovery_window_minutes : int
            Maximum window to measure recovery.

        Returns
        -------
        dict[str, float]
            Recovery metrics including median recovery time and recovery rate.
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
            # Look for recovery within window
            window_end = shock_time + pd.Timedelta(minutes=recovery_window_minutes)
            window_data = spread.loc[shock_time:window_end]

            if len(window_data) < 2:
                continue

            # Find first return to median
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

    def compute_spread_curve(
        self, df: pd.DataFrame, resample_freq: str = "5min"
    ) -> pd.DataFrame:
        """
        Compute spread tightening curve.

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
        self, df: pd.DataFrame, percentiles: list[float] | None = None
    ) -> dict[str, float]:
        """
        Compute spread percentile statistics.

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

    def compute_spread_regime_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute spread statistics by regime.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with spread and regime data.

        Returns
        -------
        pd.DataFrame
            Spread statistics by regime.
        """
        if "spread" not in df.columns or "regime" not in df.columns:
            return pd.DataFrame()

        return df.groupby("regime")["spread"].agg([
            "mean", "std", "median", "count"
        ])


class TrajectoryAnalyzer:
    """
    Analyzes event-aligned trajectories.

    Computes normalized trajectories and median paths
    aligned to event times.
    """

    def __init__(
        self,
        pre_event_minutes: int = 60,
        post_event_minutes: int = 30,
    ):
        self.pre_event_minutes = pre_event_minutes
        self.post_event_minutes = post_event_minutes

    def extract_event_window(
        self,
        contract: ContractTimeseries,
        event_time: pd.Timestamp | datetime,
    ) -> EventWindow | None:
        """
        Extract data window around event time.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract data.
        event_time : pd.Timestamp | datetime
            Event timestamp.

        Returns
        -------
        EventWindow | None
            Event window or None if insufficient data.
        """
        if isinstance(event_time, datetime):
            event_time = pd.Timestamp(event_time)

        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("UTC")

        pre_start = event_time - timedelta(minutes=self.pre_event_minutes)
        post_end = event_time + timedelta(minutes=self.post_event_minutes)

        df = contract.data

        pre_mask = (df.index >= pre_start) & (df.index < event_time)
        post_mask = (df.index >= event_time) & (df.index <= post_end)

        pre_data = df[pre_mask].copy()
        post_data = df[post_mask].copy()

        if pre_data.empty and post_data.empty:
            return None

        return EventWindow(
            contract_id=contract.contract_id,
            event_time=event_time,
            pre_event_data=pre_data,
            post_event_data=post_data,
        )

    def normalize_trajectory(
        self,
        df: pd.DataFrame,
        event_time: pd.Timestamp,
        column: str = "price_c",
    ) -> pd.DataFrame:
        """
        Normalize trajectory relative to event time.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        event_time : pd.Timestamp
            Reference event time.
        column : str
            Column to normalize.

        Returns
        -------
        pd.DataFrame
            Normalized trajectory with minutes_to_event index.
        """
        if column not in df.columns:
            return pd.DataFrame()

        df = df.copy()

        df["minutes_to_event"] = (df.index - event_time).total_seconds() / 60

        anchor_value = df.loc[df["minutes_to_event"].abs().idxmin(), column]
        if pd.isna(anchor_value) or anchor_value == 0:
            anchor_value = df[column].iloc[0]

        df["normalized"] = df[column] / anchor_value

        return df[["minutes_to_event", column, "normalized"]].set_index("minutes_to_event")

    def compute_median_trajectory(
        self,
        trajectories: list[pd.DataFrame],
        column: str = "normalized",
    ) -> pd.DataFrame:
        """
        Compute median trajectory across multiple contracts.

        Parameters
        ----------
        trajectories : list[pd.DataFrame]
            List of normalized trajectories.
        column : str
            Column to aggregate.

        Returns
        -------
        pd.DataFrame
            Median trajectory with confidence bands.
        """
        if not trajectories:
            return pd.DataFrame()

        combined = pd.concat([
            t[[column]].rename(columns={column: f"traj_{i}"})
            for i, t in enumerate(trajectories)
        ], axis=1)

        result = pd.DataFrame(index=combined.index)
        result["median"] = combined.median(axis=1)
        result["q25"] = combined.quantile(0.25, axis=1)
        result["q75"] = combined.quantile(0.75, axis=1)
        result["std"] = combined.std(axis=1)
        result["count"] = combined.notna().sum(axis=1)

        return result


class DepthAnalyzer:
    """
    Analyzes order book depth dynamics.

    Computes depth resilience and recovery metrics.
    """

    def compute_depth_resilience(
        self, df: pd.DataFrame, window: int = 10
    ) -> pd.Series:
        """
        Compute depth resilience metric.

        Measures how quickly depth recovers after large trades.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with spread data.
        window : int
            Analysis window.

        Returns
        -------
        pd.Series
            Depth resilience scores.
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

    def compute_depth_impact(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute volume-weighted depth impact.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with volume and spread data.

        Returns
        -------
        pd.DataFrame
            Depth impact metrics.
        """
        if "volume" not in df.columns or "spread" not in df.columns:
            return pd.DataFrame()

        result = pd.DataFrame(index=df.index)

        volume_ma = df["volume"].rolling(window=20, min_periods=1).mean()
        spread_ma = df["spread"].rolling(window=20, min_periods=1).mean()

        result["volume_spread_corr"] = df["volume"].rolling(window=60).corr(df["spread"])

        large_volume_mask = df["volume"] > 2 * volume_ma
        result["post_large_spread"] = np.where(
            large_volume_mask.shift(1).fillna(False),
            df["spread"],
            np.nan,
        )

        return result


class PrePostAnalyzer:
    """
    Analyzes pre vs post event differences.

    Computes statistical comparisons of market behavior
    before and after events.
    """

    def compare_periods(
        self,
        pre_data: pd.DataFrame,
        post_data: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Compare metrics between pre and post periods.

        Parameters
        ----------
        pre_data : pd.DataFrame
            Pre-event data.
        post_data : pd.DataFrame
            Post-event data.
        metrics : list[str] | None
            Metrics to compare.

        Returns
        -------
        dict[str, dict]
            Comparison results by metric.
        """
        if metrics is None:
            metrics = ["spread", "volume", "volatility_short"]

        results = {}

        for metric in metrics:
            if metric not in pre_data.columns or metric not in post_data.columns:
                continue

            pre_values = pre_data[metric].dropna()
            post_values = post_data[metric].dropna()

            if len(pre_values) < 3 or len(post_values) < 3:
                continue

            try:
                t_stat, t_pvalue = stats.ttest_ind(pre_values, post_values)
            except Exception:
                t_stat, t_pvalue = np.nan, np.nan

            try:
                ks_stat, ks_pvalue = stats.ks_2samp(pre_values, post_values)
            except Exception:
                ks_stat, ks_pvalue = np.nan, np.nan

            results[metric] = {
                "pre_mean": pre_values.mean(),
                "post_mean": post_values.mean(),
                "pre_std": pre_values.std(),
                "post_std": post_values.std(),
                "change_pct": (post_values.mean() - pre_values.mean()) / pre_values.mean()
                if pre_values.mean() != 0 else np.nan,
                "t_statistic": t_stat,
                "t_pvalue": t_pvalue,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
            }

        return results


class MicrostructureAnalyzer:
    """
    Main microstructure analysis orchestrator.

    Coordinates all microstructure analytics and produces
    comprehensive market analysis.
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize microstructure analyzer.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration. Uses global if None.
        """
        self.config = config or get_config()
        micro_config = self.config.microstructure

        self.surge_detector = SurgeDetector(
            volume_threshold=micro_config.surge_detection.volume_threshold,
            volatility_threshold=micro_config.surge_detection.volatility_threshold,
            lookback_window=micro_config.surge_detection.lookback_window,
        )
        self.spread_analyzer = SpreadAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer(
            pre_event_minutes=micro_config.event_alignment.pre_event_minutes,
            post_event_minutes=micro_config.event_alignment.post_event_minutes,
        )
        self.depth_analyzer = DepthAnalyzer()
        self.prepost_analyzer = PrePostAnalyzer()

    def analyze_contract(
        self, contract: ContractTimeseries
    ) -> MicrostructureMetrics:
        """
        Analyze single contract microstructure.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to analyze.

        Returns
        -------
        MicrostructureMetrics
            Microstructure metrics.
        """
        df = contract.data

        volume_surges = self.surge_detector.detect_volume_surges(
            df, contract.contract_id
        )
        volatility_surges = self.surge_detector.detect_volatility_surges(
            df, contract.contract_id
        )

        spread_stats = self.spread_analyzer.compute_spread_percentiles(df)

        # Robust spread collapse slope (Section 5)
        spread_collapse = self.spread_analyzer.robust_spread_collapse_slope(df)

        # Liquidity resilience (Section 6)
        liquidity_resilience = self.spread_analyzer.compute_liquidity_resilience(df)

        avg_spread = df["spread"].mean() if "spread" in df.columns else np.nan
        avg_spread_pct = df["spread_pct"].mean() if "spread_pct" in df.columns else np.nan
        avg_volume = df["volume"].mean() if "volume" in df.columns else np.nan

        volatility = df["volatility_short"].mean() if "volatility_short" in df.columns else np.nan

        depth_resilience = self.depth_analyzer.compute_depth_resilience(df)
        avg_resilience = depth_resilience.mean() if not depth_resilience.empty else np.nan

        liquidity_score = self._compute_liquidity_score(df)

        metrics = {
            **spread_stats,
            **spread_collapse,  # Section 5
            **liquidity_resilience,  # Section 6
            "volume_surge_count": len(volume_surges),
            "volatility_surge_count": len(volatility_surges),
        }

        return MicrostructureMetrics(
            contract_id=contract.contract_id,
            category=contract.category,
            avg_spread=avg_spread,
            avg_spread_pct=avg_spread_pct,
            avg_volume=avg_volume,
            volatility=volatility,
            liquidity_score=liquidity_score,
            depth_resilience=avg_resilience,
            surge_count=len(volume_surges) + len(volatility_surges),
            metrics=metrics,
        )

    def analyze_dataset(
        self, dataset: MarketDataset
    ) -> tuple[list[MicrostructureMetrics], pd.DataFrame]:
        """
        Analyze entire dataset microstructure.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to analyze.

        Returns
        -------
        tuple[list[MicrostructureMetrics], pd.DataFrame]
            (List of metrics, summary DataFrame)
        """
        logger.info(f"Analyzing microstructure for {len(dataset)} contracts")

        all_metrics = []
        for contract in dataset:
            metrics = self.analyze_contract(contract)
            all_metrics.append(metrics)

        summary_df = self._metrics_to_dataframe(all_metrics)

        logger.info("Microstructure analysis complete")
        return all_metrics, summary_df

    def compute_category_aggregates(
        self, dataset: MarketDataset
    ) -> pd.DataFrame:
        """
        Compute aggregate microstructure stats by category.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to analyze.

        Returns
        -------
        pd.DataFrame
            Category-level aggregate statistics.
        """
        records = []

        for category in dataset.categories:
            cat_contracts = dataset.filter_by_category(category)

            spreads = []
            volumes = []
            volatilities = []

            for contract in cat_contracts:
                df = contract.data
                if "spread" in df.columns:
                    spreads.extend(df["spread"].dropna().tolist())
                if "volume" in df.columns:
                    volumes.extend(df["volume"].dropna().tolist())
                if "volatility_short" in df.columns:
                    volatilities.extend(df["volatility_short"].dropna().tolist())

            records.append({
                "category": category,
                "n_contracts": len(cat_contracts),
                "avg_spread": np.mean(spreads) if spreads else np.nan,
                "median_spread": np.median(spreads) if spreads else np.nan,
                "avg_volume": np.mean(volumes) if volumes else np.nan,
                "total_volume": np.sum(volumes) if volumes else 0,
                "avg_volatility": np.mean(volatilities) if volatilities else np.nan,
            })

        return pd.DataFrame(records)

    def _compute_liquidity_score(self, df: pd.DataFrame) -> float:
        """
        Compute composite liquidity score.

        Combines multiple liquidity indicators into single score.
        """
        components = []

        if "volume" in df.columns:
            vol_zscore = (df["volume"] - df["volume"].mean()) / df["volume"].std()
            components.append(vol_zscore.mean())

        if "spread" in df.columns:
            spread_inv = 1 / df["spread"].replace(0, np.nan)
            spread_zscore = (spread_inv - spread_inv.mean()) / spread_inv.std()
            components.append(spread_zscore.mean())

        if not components:
            return np.nan

        return np.mean(components)

    def _metrics_to_dataframe(
        self, metrics: list[MicrostructureMetrics]
    ) -> pd.DataFrame:
        """Convert metrics list to DataFrame."""
        records = []
        for m in metrics:
            record = {
                "contract_id": m.contract_id,
                "category": m.category,
                "avg_spread": m.avg_spread,
                "avg_spread_pct": m.avg_spread_pct,
                "avg_volume": m.avg_volume,
                "volatility": m.volatility,
                "liquidity_score": m.liquidity_score,
                "depth_resilience": m.depth_resilience,
                "surge_count": m.surge_count,
                **m.metrics,
            }
            records.append(record)

        return pd.DataFrame(records)


def analyze_microstructure(
    dataset: MarketDataset, config: IRPConfig | None = None
) -> tuple[list[MicrostructureMetrics], pd.DataFrame]:
    """
    Convenience function for microstructure analysis.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to analyze.
    config : IRPConfig | None
        Platform configuration.

    Returns
    -------
    tuple[list[MicrostructureMetrics], pd.DataFrame]
        (Metrics list, summary DataFrame)
    """
    analyzer = MicrostructureAnalyzer(config)
    return analyzer.analyze_dataset(dataset)
