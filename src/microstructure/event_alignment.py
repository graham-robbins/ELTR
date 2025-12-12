"""
Event-aligned trajectory analysis for IRP platform.

Provides event-time normalization and trajectory computation for
cross-contract comparison with aligned temporal axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy import interpolate

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    Category,
    ContractID,
    ContractTimeseries,
    MarketDataset,
)
from src.utils.normalization import normalize_series
from src.utils.time_binning import detect_gaps

logger = get_logger("event_alignment")

# Default gap threshold for masking interpolated values (Section 7)
DEFAULT_GAP_THRESHOLD_MINUTES = 10.0


@dataclass
class AlignedTrajectory:
    """Container for a single event-aligned trajectory."""
    contract_id: ContractID
    category: Category
    event_time: pd.Timestamp
    time_grid: np.ndarray
    values: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)


@dataclass
class AggregatedTrajectory:
    """Container for aggregated trajectory statistics."""
    category: Category | None
    metric: str
    time_grid: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    p10: np.ndarray
    p25: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    count: np.ndarray
    n_contracts: int


class EventAligner:
    """
    Aligns contract timeseries to event time.

    Produces normalized trajectories for cross-contract comparison
    on a common time grid relative to event occurrence.
    """

    def __init__(
        self,
        grid_hours_pre: float = 24.0,
        grid_hours_post: float = 4.0,
        grid_resolution_minutes: float = 1.0,
        interpolation_method: str = "linear",
        gap_threshold_minutes: float = DEFAULT_GAP_THRESHOLD_MINUTES,
    ):
        """
        Initialize event aligner.

        Parameters
        ----------
        grid_hours_pre : float
            Hours before event to include.
        grid_hours_post : float
            Hours after event to include.
        grid_resolution_minutes : float
            Time grid resolution in minutes.
        interpolation_method : str
            Interpolation method (linear, nearest, cubic).
        gap_threshold_minutes : float
            Gap threshold for masking interpolated values (Section 7).
            Interpolated values spanning gaps > threshold will be masked as NaN.
        """
        self.grid_hours_pre = grid_hours_pre
        self.grid_hours_post = grid_hours_post
        self.grid_resolution_minutes = grid_resolution_minutes
        self.interpolation_method = interpolation_method
        self.gap_threshold_minutes = gap_threshold_minutes

        self._time_grid = self._create_time_grid()

    def _create_time_grid(self) -> np.ndarray:
        """Create common time grid in minutes relative to event."""
        start_minutes = -self.grid_hours_pre * 60
        end_minutes = self.grid_hours_post * 60
        n_points = int((end_minutes - start_minutes) / self.grid_resolution_minutes) + 1
        return np.linspace(start_minutes, end_minutes, n_points)

    @property
    def time_grid(self) -> np.ndarray:
        """Get time grid in minutes relative to event."""
        return self._time_grid

    @property
    def time_grid_hours(self) -> np.ndarray:
        """Get time grid in hours relative to event."""
        return self._time_grid / 60

    def align_contract(
        self,
        contract: ContractTimeseries,
        event_time: pd.Timestamp | datetime | None = None,
        metrics: list[str] | None = None,
    ) -> AlignedTrajectory | None:
        """
        Align single contract to event time.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to align.
        event_time : pd.Timestamp | datetime | None
            Event time. Uses end of data if None.
        metrics : list[str] | None
            Metrics to interpolate. Defaults to common metrics.

        Returns
        -------
        AlignedTrajectory | None
            Aligned trajectory or None if insufficient data.
        """
        if event_time is None:
            event_time = contract.data.index.max()

        if isinstance(event_time, datetime):
            event_time = pd.Timestamp(event_time)

        if event_time.tzinfo is None and contract.data.index.tzinfo is not None:
            event_time = event_time.tz_localize(contract.data.index.tzinfo)

        if metrics is None:
            metrics = self._get_default_metrics(contract.data)

        df = contract.data.copy()
        df["minutes_to_event"] = (event_time - df.index).total_seconds() / 60

        values = {}
        for metric in metrics:
            if metric not in df.columns:
                continue

            interpolated = self._interpolate_to_grid(
                df["minutes_to_event"].values,
                df[metric].values,
            )

            if interpolated is not None:
                values[metric] = interpolated

        if not values:
            return None

        return AlignedTrajectory(
            contract_id=contract.contract_id,
            category=contract.category,
            event_time=event_time,
            time_grid=self._time_grid.copy(),
            values=values,
            metadata={
                "n_original_obs": len(df),
                "data_start": df.index.min(),
                "data_end": df.index.max(),
            },
        )

    def _get_default_metrics(self, df: pd.DataFrame) -> list[str]:
        """Get default metrics to align."""
        candidates = [
            "midpoint", "price_c", "spread", "spread_pct",
            "volume", "volatility_short", "depth_resilience",
            "book_thinning", "volume_zscore",
        ]
        return [m for m in candidates if m in df.columns]

    def _interpolate_to_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray | None:
        """
        Interpolate values to common time grid.

        Masks interpolated values that span gaps > threshold (Section 7).
        """
        mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[mask]
        y_valid = y[mask]

        if len(x_valid) < 3:
            return None

        x_sorted_idx = np.argsort(x_valid)
        x_sorted = x_valid[x_sorted_idx]
        y_sorted = y_valid[x_sorted_idx]

        grid_mask = (self._time_grid >= x_sorted.min()) & (self._time_grid <= x_sorted.max())

        try:
            if self.interpolation_method == "linear":
                f = interpolate.interp1d(
                    x_sorted, y_sorted,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )
            elif self.interpolation_method == "nearest":
                f = interpolate.interp1d(
                    x_sorted, y_sorted,
                    kind="nearest",
                    fill_value=np.nan,
                    bounds_error=False,
                )
            elif self.interpolation_method == "cubic":
                f = interpolate.interp1d(
                    x_sorted, y_sorted,
                    kind="cubic",
                    fill_value=np.nan,
                    bounds_error=False,
                )
            else:
                f = interpolate.interp1d(
                    x_sorted, y_sorted,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )

            result = np.full_like(self._time_grid, np.nan)
            result[grid_mask] = f(self._time_grid[grid_mask])

            # Mask interpolated values spanning long gaps (Section 7)
            result = self._mask_long_gaps(result, x_sorted)

            return result

        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return None

    def _mask_long_gaps(
        self,
        interpolated: np.ndarray,
        x_sorted: np.ndarray,
    ) -> np.ndarray:
        """
        Mask interpolated values that span gaps > threshold (Section 7).

        Parameters
        ----------
        interpolated : np.ndarray
            Interpolated values on the time grid.
        x_sorted : np.ndarray
            Sorted original x values (minutes to event).

        Returns
        -------
        np.ndarray
            Interpolated values with gap regions masked as NaN.
        """
        if len(x_sorted) < 2:
            return interpolated

        # Find gaps between consecutive observations
        gaps = np.diff(x_sorted)
        gap_mask = gaps > self.gap_threshold_minutes

        if not gap_mask.any():
            return interpolated

        result = interpolated.copy()

        # For each gap, mask grid points that fall within it
        for i, is_gap in enumerate(gap_mask):
            if is_gap:
                gap_start = x_sorted[i]
                gap_end = x_sorted[i + 1]

                # Mask grid points within this gap (excluding endpoints)
                in_gap = (self._time_grid > gap_start) & (self._time_grid < gap_end)
                result[in_gap] = np.nan

        return result


class TrajectoryAggregator:
    """
    Aggregates aligned trajectories across contracts.

    Computes median trajectories and percentile bands for
    cross-contract comparison. Supports weighted aggregation
    based on observation counts (Section 11).
    """

    def __init__(
        self,
        percentiles: list[float] | None = None,
        use_weighted: bool = True,
    ):
        """
        Initialize aggregator.

        Parameters
        ----------
        percentiles : list[float] | None
            Percentiles for bands. Defaults to [10, 25, 75, 90].
        use_weighted : bool
            Use weighted aggregation based on observation counts (Section 11).
        """
        self.percentiles = percentiles or [10, 25, 75, 90]
        self.use_weighted = use_weighted

    def aggregate(
        self,
        trajectories: list[AlignedTrajectory],
        metric: str,
        category: Category | None = None,
        normalize: bool = True,
        norm_method: str = "zscore",
    ) -> AggregatedTrajectory | None:
        """
        Aggregate trajectories for a single metric.

        Uses weighted median/percentiles based on observation counts (Section 11).

        Parameters
        ----------
        trajectories : list[AlignedTrajectory]
            Aligned trajectories to aggregate.
        metric : str
            Metric to aggregate.
        category : Category | None
            Filter by category. All if None.
        normalize : bool
            Normalize values before aggregating.
        norm_method : str
            Normalization method.

        Returns
        -------
        AggregatedTrajectory | None
            Aggregated trajectory or None if no data.
        """
        if category is not None:
            trajectories = [t for t in trajectories if t.category == category]

        valid_trajs = [t for t in trajectories if metric in t.values]

        if not valid_trajs:
            return None

        time_grid = valid_trajs[0].time_grid
        n_points = len(time_grid)

        all_values = np.full((len(valid_trajs), n_points), np.nan)
        weights = np.ones(len(valid_trajs))  # Default uniform weights

        for i, traj in enumerate(valid_trajs):
            values = traj.values[metric]

            if normalize:
                values = self._normalize_trajectory(values, norm_method)

            all_values[i, :] = values

            # Extract observation count for weighting (Section 11)
            if self.use_weighted and "n_original_obs" in traj.metadata:
                weights[i] = traj.metadata["n_original_obs"]

        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        with np.errstate(all="ignore"):
            if self.use_weighted:
                # Weighted aggregation (Section 11)
                median = self._weighted_percentile_by_column(all_values, weights, 50)
                mean = self._weighted_mean_by_column(all_values, weights)
                p10 = self._weighted_percentile_by_column(all_values, weights, 10)
                p25 = self._weighted_percentile_by_column(all_values, weights, 25)
                p75 = self._weighted_percentile_by_column(all_values, weights, 75)
                p90 = self._weighted_percentile_by_column(all_values, weights, 90)
            else:
                # Unweighted aggregation
                median = np.nanmedian(all_values, axis=0)
                mean = np.nanmean(all_values, axis=0)
                p10 = np.nanpercentile(all_values, 10, axis=0)
                p25 = np.nanpercentile(all_values, 25, axis=0)
                p75 = np.nanpercentile(all_values, 75, axis=0)
                p90 = np.nanpercentile(all_values, 90, axis=0)

            std = np.nanstd(all_values, axis=0)
            count = np.sum(~np.isnan(all_values), axis=0)

        return AggregatedTrajectory(
            category=category,
            metric=metric,
            time_grid=time_grid,
            median=median,
            mean=mean,
            std=std,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            count=count,
            n_contracts=len(valid_trajs),
        )

    def _weighted_percentile_by_column(
        self,
        data: np.ndarray,
        weights: np.ndarray,
        percentile: float,
    ) -> np.ndarray:
        """
        Compute weighted percentile for each column (Section 11).

        Parameters
        ----------
        data : np.ndarray
            2D array (n_trajectories, n_time_points).
        weights : np.ndarray
            1D array of weights per trajectory.
        percentile : float
            Percentile to compute (0-100).

        Returns
        -------
        np.ndarray
            Weighted percentile for each time point.
        """
        n_cols = data.shape[1]
        result = np.full(n_cols, np.nan)

        for j in range(n_cols):
            col_data = data[:, j]
            valid_mask = ~np.isnan(col_data)

            if valid_mask.sum() == 0:
                continue

            valid_data = col_data[valid_mask]
            valid_weights = weights[valid_mask]

            # Normalize weights for this column
            w_sum = valid_weights.sum()
            if w_sum == 0:
                result[j] = np.nanmedian(valid_data)
                continue

            valid_weights = valid_weights / w_sum

            # Sort by data values
            sort_idx = np.argsort(valid_data)
            sorted_data = valid_data[sort_idx]
            sorted_weights = valid_weights[sort_idx]

            # Compute cumulative weights
            cum_weights = np.cumsum(sorted_weights)

            # Find value at percentile
            target = percentile / 100.0
            idx = np.searchsorted(cum_weights, target)
            idx = min(idx, len(sorted_data) - 1)
            result[j] = sorted_data[idx]

        return result

    def _weighted_mean_by_column(
        self,
        data: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted mean for each column (Section 11).

        Parameters
        ----------
        data : np.ndarray
            2D array (n_trajectories, n_time_points).
        weights : np.ndarray
            1D array of weights per trajectory.

        Returns
        -------
        np.ndarray
            Weighted mean for each time point.
        """
        n_cols = data.shape[1]
        result = np.full(n_cols, np.nan)

        for j in range(n_cols):
            col_data = data[:, j]
            valid_mask = ~np.isnan(col_data)

            if valid_mask.sum() == 0:
                continue

            valid_data = col_data[valid_mask]
            valid_weights = weights[valid_mask]

            w_sum = valid_weights.sum()
            if w_sum == 0:
                result[j] = np.nanmean(valid_data)
            else:
                result[j] = np.sum(valid_data * valid_weights) / w_sum

        return result

    def _normalize_trajectory(
        self,
        values: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Normalize a single trajectory."""
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 2:
            return values

        valid_values = values[valid_mask]

        if method == "zscore":
            mean = np.nanmean(valid_values)
            std = np.nanstd(valid_values)
            if std > 0:
                result = (values - mean) / std
            else:
                result = np.zeros_like(values)

        elif method == "minmax":
            min_val = np.nanmin(valid_values)
            max_val = np.nanmax(valid_values)
            if max_val > min_val:
                result = (values - min_val) / (max_val - min_val)
            else:
                result = np.full_like(values, 0.5)

        elif method == "anchor":
            anchor_idx = len(values) // 2
            if not np.isnan(values[anchor_idx]):
                anchor = values[anchor_idx]
                if anchor != 0:
                    result = values / anchor
                else:
                    result = values
            else:
                result = values

        else:
            result = values

        return result

    def aggregate_all_metrics(
        self,
        trajectories: list[AlignedTrajectory],
        metrics: list[str] | None = None,
        by_category: bool = True,
        normalize: bool = True,
    ) -> dict[str, list[AggregatedTrajectory]]:
        """
        Aggregate multiple metrics, optionally by category.

        Parameters
        ----------
        trajectories : list[AlignedTrajectory]
            Trajectories to aggregate.
        metrics : list[str] | None
            Metrics to aggregate.
        by_category : bool
            Separate aggregations by category.
        normalize : bool
            Normalize before aggregating.

        Returns
        -------
        dict[str, list[AggregatedTrajectory]]
            Mapping from metric to list of aggregated trajectories.
        """
        if metrics is None:
            all_metrics = set()
            for traj in trajectories:
                all_metrics.update(traj.values.keys())
            metrics = list(all_metrics)

        categories = [None]
        if by_category:
            cat_set = set(t.category for t in trajectories)
            categories = list(cat_set) + [None]

        results = {}
        for metric in metrics:
            metric_results = []
            for category in categories:
                agg = self.aggregate(
                    trajectories, metric,
                    category=category,
                    normalize=normalize,
                )
                if agg is not None:
                    metric_results.append(agg)
            results[metric] = metric_results

        return results


class EventTrajectoryAnalyzer:
    """
    Main event trajectory analysis orchestrator.

    Coordinates alignment and aggregation to produce
    signature trajectory plots for the IRP.
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize analyzer.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration.
        """
        self.config = config or get_config()

        event_config = self.config.microstructure.event_alignment
        self.aligner = EventAligner(
            grid_hours_pre=event_config.grid_hours_pre,
            grid_hours_post=event_config.grid_hours_post,
            grid_resolution_minutes=event_config.interpolation_grid_minutes,
        )
        self.aggregator = TrajectoryAggregator()

    def analyze_dataset(
        self,
        dataset: MarketDataset,
        metrics: list[str] | None = None,
    ) -> dict[str, list[AggregatedTrajectory]]:
        """
        Analyze entire dataset for event trajectories.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to analyze.
        metrics : list[str] | None
            Metrics to analyze.

        Returns
        -------
        dict[str, list[AggregatedTrajectory]]
            Aggregated trajectories by metric.
        """
        logger.info(f"Aligning {len(dataset)} contracts to event time")

        trajectories = []
        for contract in dataset:
            aligned = self.aligner.align_contract(contract, metrics=metrics)
            if aligned is not None:
                trajectories.append(aligned)

        logger.info(f"Successfully aligned {len(trajectories)} contracts")

        if not trajectories:
            return {}

        results = self.aggregator.aggregate_all_metrics(
            trajectories,
            metrics=metrics,
            by_category=True,
            normalize=True,
        )

        return results

    def compute_signature_trajectories(
        self,
        dataset: MarketDataset,
    ) -> dict[str, AggregatedTrajectory]:
        """
        Compute signature trajectories for IRP.

        Produces median trajectories for:
        - Midprice evolution
        - Spread tightening
        - Depth dynamics
        - Volatility ramp

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to analyze.

        Returns
        -------
        dict[str, AggregatedTrajectory]
            Signature trajectories.
        """
        signature_metrics = [
            "midpoint",
            "spread",
            "spread_pct",
            "volatility_short",
            "volume_zscore",
            "depth_resilience",
        ]

        all_results = self.analyze_dataset(dataset, metrics=signature_metrics)

        signatures = {}
        for metric, agg_list in all_results.items():
            overall = [a for a in agg_list if a.category is None]
            if overall:
                signatures[f"{metric}_overall"] = overall[0]

            for agg in agg_list:
                if agg.category is not None:
                    signatures[f"{metric}_{agg.category}"] = agg

        return signatures

    def get_time_grid_hours(self) -> np.ndarray:
        """Get time grid in hours for plotting."""
        return self.aligner.time_grid_hours


def compute_event_trajectories(
    dataset: MarketDataset,
    config: IRPConfig | None = None,
) -> dict[str, list[AggregatedTrajectory]]:
    """
    Convenience function for event trajectory analysis.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to analyze.
    config : IRPConfig | None
        Platform configuration.

    Returns
    -------
    dict[str, list[AggregatedTrajectory]]
        Aggregated trajectories.
    """
    analyzer = EventTrajectoryAnalyzer(config)
    return analyzer.analyze_dataset(dataset)
