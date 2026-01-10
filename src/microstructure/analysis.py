"""
Microstructure Analysis Orchestration.

This module provides high-level functions that combine metrics from:
    - spread.py: Bid-ask spread computation
    - liquidity.py: Depth and resilience metrics
    - bursts.py: Surge and burst detection
    - regimes.py: State classification and transitions
    - lifecycle.py: Contract lifecycle normalization

Usage:
    from src.microstructure.analysis import run_full_analysis
    results = run_full_analysis(df, config)

AUTHOR MUST DOCUMENT:
    - Order of operations and dependencies between metrics
    - Default configuration values and their justification
    - Data requirements and minimum observation thresholds

Note:
    This module maintains backward compatibility with the original analysis.py
    by re-exporting classes and functions from the new modular structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

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

# Import from refactored modules
from src.microstructure.spread import (
    compute_spread_features,
    compute_spread_collapse_slope,
    compute_spread_percentiles,
    compute_spread_curve,
    compute_spread_regime_stats,
)
from src.microstructure.liquidity import (
    compute_volume_features,
    compute_depth_features,
    compute_depth_resilience,
    compute_depth_impact,
    compute_liquidity_resilience,
)
from src.microstructure.bursts import (
    SurgeEvent,
    detect_volume_surges,
    detect_volatility_surges,
    compute_surge_ratio,
    classify_volatility_burst,
    compute_burst_intensity,
)
from src.microstructure.regimes import (
    MicrostructureState,
    compute_event_regime,
    compute_microstructure_regime,
    compute_transition_matrix,
    compute_regime_entropy,
)
from src.microstructure.lifecycle import (
    LIFECYCLE_BIN_EDGES,
    LIFECYCLE_PHASE_NAMES,
    compute_lifecycle_features,
    make_lifecycle_bins,
    assign_lifecycle_bins,
    compute_binned_trajectory,
)

logger = get_logger("microstructure.analysis")


# Data structures

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
    """
    Container for microstructure summary metrics.

    AUTHOR MUST DEFINE FORMALLY: metric definitions,
    aggregation methods, interpretation guidelines
    """
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


# Legacy class wrappers (for backward compatibility)

class SurgeDetector:
    """
    Detects liquidity and volatility surges.

    BACKWARD COMPATIBILITY: This class wraps functions from bursts.py
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
        """Detect volume surge events."""
        return detect_volume_surges(
            df, contract_id, self.volume_threshold, self.lookback_window
        )

    def detect_volatility_surges(
        self, df: pd.DataFrame, contract_id: ContractID
    ) -> list[SurgeEvent]:
        """Detect volatility surge events."""
        return detect_volatility_surges(
            df, contract_id, self.volatility_threshold, self.lookback_window
        )

    def compute_surge_ratio(self, df: pd.DataFrame, metric: str = "volume") -> pd.Series:
        """Compute rolling surge ratio."""
        return compute_surge_ratio(df, metric, self.lookback_window)


class SpreadAnalyzer:
    """
    Analyzes bid-ask spread dynamics.

    BACKWARD COMPATIBILITY: This class wraps functions from spread.py and liquidity.py
    """

    def robust_spread_collapse_slope(
        self, df: pd.DataFrame, lifecycle_col: str = "lifecycle_ratio"
    ) -> dict[str, float]:
        """Fast slope estimate of log(spread) over lifecycle."""
        return compute_spread_collapse_slope(df, lifecycle_col)

    def compute_liquidity_resilience(
        self,
        df: pd.DataFrame,
        shock_threshold_quantile: float = 0.95,
        recovery_window_minutes: int = 30,
    ) -> dict[str, float]:
        """Compute liquidity resilience via spread-shock recovery time."""
        return compute_liquidity_resilience(
            df, shock_threshold_quantile, recovery_window_minutes
        )

    def compute_spread_curve(
        self, df: pd.DataFrame, resample_freq: str = "5min"
    ) -> pd.DataFrame:
        """Compute spread tightening curve."""
        return compute_spread_curve(df, resample_freq)

    def compute_spread_percentiles(
        self, df: pd.DataFrame, percentiles: list[float] | None = None
    ) -> dict[str, float]:
        """Compute spread percentile statistics."""
        return compute_spread_percentiles(df, percentiles)

    def compute_spread_regime_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute spread statistics by regime."""
        return compute_spread_regime_stats(df)


class DepthAnalyzer:
    """
    Analyzes order book depth dynamics.

    BACKWARD COMPATIBILITY: This class wraps functions from liquidity.py
    """

    def compute_depth_resilience(
        self, df: pd.DataFrame, window: int = 10
    ) -> pd.Series:
        """Compute depth resilience metric."""
        return compute_depth_resilience(df, window)

    def compute_depth_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-weighted depth impact."""
        return compute_depth_impact(df)


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
        """Extract data window around event time."""
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
        """Normalize trajectory relative to event time."""
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
        """Compute median trajectory across multiple contracts."""
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
        """Compare metrics between pre and post periods."""
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


# Main orchestrator

class MicrostructureAnalyzer:
    """
    Main microstructure analysis orchestrator.

    Coordinates all microstructure analytics and produces
    comprehensive market analysis.

    Parameters
    ----------
    config : IRPConfig | None
        Platform configuration. Uses global if None.
    n_jobs : int
        Number of parallel workers for dataset analysis.
        Use -1 for all available cores, 1 for sequential processing.
        Default uses config.pipeline.parallel_workers.
    """

    def __init__(self, config: IRPConfig | None = None, n_jobs: int | None = None):
        """Initialize microstructure analyzer."""
        self.config = config or get_config()
        micro_config = self.config.microstructure

        # Set parallel workers: parameter > config > default
        if n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = self.config.pipeline.parallel_workers

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
        """Analyze single contract microstructure."""
        df = contract.data

        volume_surges = self.surge_detector.detect_volume_surges(
            df, contract.contract_id
        )
        volatility_surges = self.surge_detector.detect_volatility_surges(
            df, contract.contract_id
        )

        spread_stats = self.spread_analyzer.compute_spread_percentiles(df)
        spread_collapse = self.spread_analyzer.robust_spread_collapse_slope(df)
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
            **spread_collapse,
            **liquidity_resilience,
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

        Uses parallel processing when joblib is available and n_jobs != 1.
        Falls back to sequential processing otherwise. Individual contract
        failures are logged and skipped rather than failing the entire batch.
        """
        logger.info(f"Analyzing microstructure for {len(dataset)} contracts")

        contracts = list(dataset)

        # Use parallel processing if available and configured
        if JOBLIB_AVAILABLE and self.n_jobs != 1 and len(contracts) > 1:
            logger.info(f"Using parallel processing with {self.n_jobs} workers")
            all_metrics = Parallel(n_jobs=self.n_jobs)(
                delayed(self._safe_analyze_contract)(contract)
                for contract in contracts
            )
            # Filter out None values from failed analyses
            all_metrics = [m for m in all_metrics if m is not None]
        else:
            # Sequential fallback with error handling
            all_metrics = []
            for contract in contracts:
                metrics = self._safe_analyze_contract(contract)
                if metrics is not None:
                    all_metrics.append(metrics)

        summary_df = self._metrics_to_dataframe(all_metrics)

        logger.info(f"Microstructure analysis complete ({len(all_metrics)}/{len(contracts)} contracts)")
        return all_metrics, summary_df

    def _safe_analyze_contract(
        self, contract: ContractTimeseries
    ) -> MicrostructureMetrics | None:
        """
        Safely analyze a single contract, catching and logging errors.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to analyze.

        Returns
        -------
        MicrostructureMetrics | None
            Metrics if successful, None if analysis failed.
        """
        try:
            return self.analyze_contract(contract)
        except Exception as e:
            logger.warning(
                f"Failed to analyze contract {contract.contract_id}: {e}"
            )
            return None

    def compute_category_aggregates(
        self, dataset: MarketDataset
    ) -> pd.DataFrame:
        """Compute aggregate microstructure stats by category."""
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
        """Compute composite liquidity score."""
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


# Convenience functions

def analyze_microstructure(
    dataset: MarketDataset, config: IRPConfig | None = None
) -> tuple[list[MicrostructureMetrics], pd.DataFrame]:
    """Convenience function for microstructure analysis."""
    analyzer = MicrostructureAnalyzer(config)
    return analyzer.analyze_dataset(dataset)


def run_full_analysis(
    df: pd.DataFrame,
    contract_id: str = "unknown",
    config: IRPConfig | None = None,
) -> dict[str, any]:
    """
    Run full microstructure analysis on a single DataFrame.

    Convenience function that runs all analysis modules and returns
    combined results.
    """
    results = {}

    # Spread metrics
    results["spread_percentiles"] = compute_spread_percentiles(df)
    results["spread_collapse"] = compute_spread_collapse_slope(df)
    results["liquidity_resilience"] = compute_liquidity_resilience(df)

    # Burst metrics
    results["volume_surges"] = len(detect_volume_surges(df, contract_id))
    results["volatility_surges"] = len(detect_volatility_surges(df, contract_id))
    results["burst_intensity"] = compute_burst_intensity(df)

    # Regime metrics
    results["regime_entropy"] = compute_regime_entropy(df)

    # Depth resilience
    depth_res = compute_depth_resilience(df)
    results["avg_depth_resilience"] = depth_res.mean() if len(depth_res) > 0 else np.nan

    return results


# Backward compatibility exports

__all__ = [
    # Data structures
    "SurgeEvent",
    "EventWindow",
    "MicrostructureMetrics",
    # Classes
    "SurgeDetector",
    "SpreadAnalyzer",
    "TrajectoryAnalyzer",
    "DepthAnalyzer",
    "PrePostAnalyzer",
    "MicrostructureAnalyzer",
    # Functions
    "analyze_microstructure",
    "run_full_analysis",
    # Re-exports from modules
    "MicrostructureState",
    "compute_spread_features",
    "compute_spread_collapse_slope",
    "compute_spread_percentiles",
    "compute_volume_features",
    "compute_depth_features",
    "compute_depth_resilience",
    "compute_liquidity_resilience",
    "detect_volume_surges",
    "detect_volatility_surges",
    "compute_burst_intensity",
    "compute_event_regime",
    "compute_microstructure_regime",
    "compute_transition_matrix",
    "compute_regime_entropy",
    "compute_lifecycle_features",
    "compute_binned_trajectory",
]
