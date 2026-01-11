"""
Metrics export module for IRP platform.

Provides comprehensive CSV export functionality for all
microstructure metrics, regime statistics, and trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import ContractTimeseries, MarketDataset
from src.utils.time_binning import (
    compute_lifecycle_features,
    make_lifecycle_bins,
    aggregate_by_bins,
)
from src.utils.normalization import normalize_series

logger = get_logger("export")


def sanitize_output_path(output_path: Path | str, base_dir: Path | None = None) -> Path:
    """
    Sanitize output path to prevent path traversal attacks.

    Ensures output paths don't escape their expected directory via
    directory traversal sequences.

    Parameters
    ----------
    output_path : Path | str
        User-provided output path.
    base_dir : Path | None
        Expected base directory for outputs. If provided, validates
        that the resolved path is within this directory.

    Returns
    -------
    Path
        Sanitized output path.

    Raises
    ------
    ValueError
        If the output path would escape base_dir through path traversal.
    """
    output_path = Path(output_path)

    # If a base_dir is provided, ensure the path stays within it
    if base_dir is not None:
        base_dir = Path(base_dir).resolve()

        # Handle both absolute and relative paths
        if output_path.is_absolute():
            resolved = output_path.resolve()
        else:
            resolved = (base_dir / output_path).resolve()

        # Verify the resolved path is under base_dir
        # This is the authoritative check - it catches all traversal attempts
        # regardless of how they're encoded (../, symlinks, etc.)
        try:
            resolved.relative_to(base_dir)
        except ValueError:
            raise ValueError(
                f"Security error: Output path '{output_path}' would escape "
                f"base directory '{base_dir}'"
            )

        return resolved

    # Without a base_dir, check for obvious traversal patterns
    # but allow legitimate filenames containing ".."
    path_str = str(output_path)
    traversal_patterns = ['/../', '\\..\\', '../', '..\\']
    if path_str.startswith('..') or any(p in path_str for p in traversal_patterns):
        raise ValueError(
            f"Security error: Path traversal detected in output path: '{output_path}'"
        )

    return output_path


@dataclass
class ContractSummaryMetrics:
    """Container for contract-level summary metrics."""
    contract_id: str
    category: str
    n_observations: int
    duration_hours: float
    avg_spread: float
    avg_spread_pct: float
    avg_volume: float
    total_volume: float
    avg_volatility: float
    max_volatility: float
    avg_depth_resilience: float
    spread_collapse_slope: float
    liquidity_resilience: float
    burst_intensity: float
    regime_entropy: float
    dominant_regime: str
    regime_proportions: dict = field(default_factory=dict)
    lifecycle_metrics: dict = field(default_factory=dict)


@dataclass
class CategorySummaryMetrics:
    """Container for category-level summary metrics."""
    category: str
    n_contracts: int
    total_observations: int
    avg_spread: float
    avg_spread_pct: float
    avg_volume: float
    avg_volatility: float
    avg_depth_resilience: float
    avg_spread_collapse_slope: float
    avg_liquidity_resilience: float
    avg_burst_intensity: float
    avg_regime_entropy: float
    regime_proportions: dict = field(default_factory=dict)


class MetricsExporter:
    """
    Exports microstructure metrics to CSV files.

    Computes and exports:
    - Contract-level summary metrics
    - Category-level aggregated metrics
    - Lifecycle-binned metrics
    - Regime statistics
    - Event trajectory summaries
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize exporter.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration.
        """
        self.config = config or get_config()

    def compute_contract_metrics(
        self,
        contract: ContractTimeseries,
    ) -> ContractSummaryMetrics:
        """
        Compute summary metrics for a single contract.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to analyze.

        Returns
        -------
        ContractSummaryMetrics
            Computed metrics.
        """
        df = contract.data

        # Basic statistics
        n_obs = len(df)
        duration_hours = 0.0
        if len(df) > 1:
            duration_hours = (df.index.max() - df.index.min()).total_seconds() / 3600

        # Spread metrics
        avg_spread = df["spread"].mean() if "spread" in df.columns else np.nan
        avg_spread_pct = df["spread_pct"].mean() if "spread_pct" in df.columns else np.nan

        # Volume metrics
        avg_volume = df["volume"].mean() if "volume" in df.columns else np.nan
        total_volume = df["volume"].sum() if "volume" in df.columns else np.nan

        # Volatility metrics
        avg_volatility = np.nan
        max_volatility = np.nan
        if "volatility_short" in df.columns:
            avg_volatility = df["volatility_short"].mean()
            max_volatility = df["volatility_short"].max()

        # Depth metrics
        avg_depth = df["depth_resilience"].mean() if "depth_resilience" in df.columns else np.nan

        # Spread collapse slope (regression of spread over lifecycle)
        spread_slope = self._compute_spread_collapse_slope(df)

        # Liquidity resilience
        liquidity_resilience = self._compute_liquidity_resilience(df)

        # Burst intensity
        burst_intensity = self._compute_burst_intensity(df)

        # Regime metrics
        regime_entropy, dominant_regime, regime_props = self._compute_regime_metrics(df)

        # Lifecycle metrics
        lifecycle_metrics = self._compute_lifecycle_metrics(df)

        return ContractSummaryMetrics(
            contract_id=contract.contract_id,
            category=contract.category,
            n_observations=n_obs,
            duration_hours=duration_hours,
            avg_spread=avg_spread,
            avg_spread_pct=avg_spread_pct,
            avg_volume=avg_volume,
            total_volume=total_volume,
            avg_volatility=avg_volatility,
            max_volatility=max_volatility,
            avg_depth_resilience=avg_depth,
            spread_collapse_slope=spread_slope,
            liquidity_resilience=liquidity_resilience,
            burst_intensity=burst_intensity,
            regime_entropy=regime_entropy,
            dominant_regime=dominant_regime,
            regime_proportions=regime_props,
            lifecycle_metrics=lifecycle_metrics,
        )

    def _compute_spread_collapse_slope(self, df: pd.DataFrame) -> float:
        """Compute spread collapse slope over lifecycle."""
        if "spread" not in df.columns or "lifecycle_ratio" not in df.columns:
            return np.nan

        mask = ~(df["spread"].isna() | df["lifecycle_ratio"].isna())
        if mask.sum() < 10:
            return np.nan

        x = df.loc[mask, "lifecycle_ratio"].values
        y = df.loc[mask, "spread"].values

        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    def _compute_liquidity_resilience(self, df: pd.DataFrame) -> float:
        """Compute liquidity resilience score."""
        if "volume" not in df.columns:
            return np.nan

        vol_zscore = normalize_series(df["volume"], method="zscore")
        if vol_zscore is None:
            return np.nan

        negative_excursions = (vol_zscore < -1).sum()
        total_obs = len(vol_zscore.dropna())

        if total_obs == 0:
            return np.nan

        return 1.0 - (negative_excursions / total_obs)

    def _compute_burst_intensity(self, df: pd.DataFrame) -> float:
        """Compute volatility burst intensity."""
        if "volatility_short" not in df.columns:
            return np.nan

        vol = df["volatility_short"].dropna()
        if len(vol) < 10:
            return np.nan

        mean_vol = vol.mean()
        std_vol = vol.std()

        if std_vol == 0:
            return 0.0

        burst_threshold = mean_vol + 2 * std_vol
        bursts = vol[vol > burst_threshold]

        if len(bursts) == 0:
            return 0.0

        return float((bursts - mean_vol).mean() / std_vol)

    def _compute_regime_metrics(
        self,
        df: pd.DataFrame,
    ) -> tuple[float, str, dict]:
        """
        Compute regime entropy and proportions.

        Entropy is normalized by log(num_states) per Section 3.5, Eq. 19-20,
        yielding a value in [0, 1] where 1 = maximum entropy (uniform distribution).
        """
        if "microstructure_state" not in df.columns:
            return np.nan, "UNKNOWN", {}

        state_counts = df["microstructure_state"].value_counts()
        total = state_counts.sum()

        if total == 0:
            return np.nan, "UNKNOWN", {}

        proportions = (state_counts / total).to_dict()

        # Convert enum names to strings
        props_clean = {}
        for k, v in proportions.items():
            key = str(k).replace("MicrostructureState.", "")
            props_clean[key] = float(v)

        # Compute entropy (Section 3.5, Eq. 19-20: normalize by log(num_states))
        probs = np.array(list(props_clean.values()))
        probs = probs[probs > 0]
        num_states = len(probs)

        if num_states <= 1:
            # Only one state = zero entropy
            entropy = 0.0
        else:
            raw_entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(num_states)
            # Normalize to [0, 1] where 1 = uniform distribution
            entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        # Find dominant regime
        dominant = max(props_clean.keys(), key=lambda x: props_clean[x])

        return float(entropy), dominant, props_clean

    def _compute_lifecycle_metrics(self, df: pd.DataFrame) -> dict:
        """Compute lifecycle-binned metrics."""
        metrics = {}

        if "lifecycle_ratio" not in df.columns:
            return metrics

        n_phases = 4
        phase_names = ["early", "mid_early", "mid_late", "late"]

        for col in ["spread", "volume", "volatility_short"]:
            if col not in df.columns:
                continue

            for i, phase in enumerate(phase_names):
                phase_start = i / n_phases
                phase_end = (i + 1) / n_phases
                phase_data = df[(df["lifecycle_ratio"] >= phase_start) &
                               (df["lifecycle_ratio"] < phase_end)]

                if len(phase_data) > 0:
                    metrics[f"{col}_{phase}"] = float(phase_data[col].median())

        return metrics

    def export_contract_summary(
        self,
        dataset: MarketDataset,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Export contract-level summary metrics to CSV.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to export.
        output_path : Path | str | None
            Output file path. Uses config if None.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame.
        """
        logger.info(f"Computing contract metrics for {len(dataset)} contracts")

        rows = []
        for contract in dataset:
            metrics = self.compute_contract_metrics(contract)

            row = {
                "contract_id": metrics.contract_id,
                "category": metrics.category,
                "n_observations": metrics.n_observations,
                "duration_hours": metrics.duration_hours,
                "avg_spread": metrics.avg_spread,
                "avg_spread_pct": metrics.avg_spread_pct,
                "avg_volume": metrics.avg_volume,
                "total_volume": metrics.total_volume,
                "avg_volatility": metrics.avg_volatility,
                "max_volatility": metrics.max_volatility,
                "avg_depth_resilience": metrics.avg_depth_resilience,
                "spread_collapse_slope": metrics.spread_collapse_slope,
                "liquidity_resilience": metrics.liquidity_resilience,
                "burst_intensity": metrics.burst_intensity,
                "regime_entropy": metrics.regime_entropy,
                "dominant_regime": metrics.dominant_regime,
            }

            # Add regime proportions as columns
            for regime, prop in metrics.regime_proportions.items():
                row[f"regime_pct_{regime}"] = prop * 100

            # Add lifecycle metrics as columns
            for lc_metric, value in metrics.lifecycle_metrics.items():
                row[f"lifecycle_{lc_metric}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path is None:
            output_path = self.config.output.tables_path / "contract_summary_extended.csv"
        else:
            # Security: Sanitize user-provided path
            output_path = sanitize_output_path(output_path, self.config.output.tables_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported contract summary to {output_path}")
        return df

    def export_category_summary(
        self,
        dataset: MarketDataset,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Export category-level summary metrics to CSV.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to export.
        output_path : Path | str | None
            Output file path.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame.
        """
        logger.info("Computing category metrics")

        # First get all contract metrics
        contract_metrics = []
        for contract in dataset:
            contract_metrics.append(self.compute_contract_metrics(contract))

        # Aggregate by category
        categories = dataset.categories
        rows = []

        for category in categories:
            cat_metrics = [m for m in contract_metrics if m.category == category]

            if not cat_metrics:
                continue

            # Aggregate regime proportions
            all_regimes = set()
            for m in cat_metrics:
                all_regimes.update(m.regime_proportions.keys())

            regime_props = {}
            for regime in all_regimes:
                props = [m.regime_proportions.get(regime, 0) for m in cat_metrics]
                regime_props[regime] = np.mean(props)

            row = {
                "category": category,
                "n_contracts": len(cat_metrics),
                "total_observations": sum(m.n_observations for m in cat_metrics),
                "avg_spread": np.mean([m.avg_spread for m in cat_metrics]),
                "avg_spread_pct": np.mean([m.avg_spread_pct for m in cat_metrics]),
                "avg_volume": np.mean([m.avg_volume for m in cat_metrics]),
                "avg_volatility": np.mean([m.avg_volatility for m in cat_metrics]),
                "avg_depth_resilience": np.nanmean([m.avg_depth_resilience for m in cat_metrics]),
                "avg_spread_collapse_slope": np.nanmean([m.spread_collapse_slope for m in cat_metrics]),
                "avg_liquidity_resilience": np.nanmean([m.liquidity_resilience for m in cat_metrics]),
                "avg_burst_intensity": np.nanmean([m.burst_intensity for m in cat_metrics]),
                "avg_regime_entropy": np.nanmean([m.regime_entropy for m in cat_metrics]),
            }

            for regime, prop in regime_props.items():
                row[f"regime_pct_{regime}"] = prop * 100

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path is None:
            output_path = self.config.output.tables_path / "category_summary_extended.csv"
        else:
            # Security: Sanitize user-provided path
            output_path = sanitize_output_path(output_path, self.config.output.tables_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported category summary to {output_path}")
        return df

    def export_lifecycle_binned(
        self,
        dataset: MarketDataset,
        n_bins: int = 50,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Export lifecycle-binned metrics to CSV.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to export.
        n_bins : int
            Number of lifecycle bins.
        output_path : Path | str | None
            Output file path.

        Returns
        -------
        pd.DataFrame
            Binned metrics DataFrame.
        """
        logger.info(f"Computing lifecycle-binned metrics with {n_bins} bins")

        metrics_cols = ["spread", "volume", "volatility_short", "depth_resilience"]
        all_rows = []

        for category in dataset.categories:
            cat_data = dataset.filter_by_category(category)

            for bin_idx in range(n_bins):
                row = {
                    "category": category,
                    "lifecycle_bin": bin_idx,
                    "lifecycle_ratio": (bin_idx + 0.5) / n_bins,
                }

                for metric in metrics_cols:
                    values = []

                    for contract in cat_data:
                        df = contract.data.copy()
                        if metric not in df.columns:
                            continue

                        if "lifecycle_ratio" not in df.columns:
                            df = compute_lifecycle_features(
                                df,
                                start_time=df.index.min(),
                                end_time=df.index.max(),
                            )

                        df["lifecycle_bin"] = make_lifecycle_bins(df, n_bins=n_bins)
                        bin_data = df[df["lifecycle_bin"] == bin_idx][metric]
                        if len(bin_data) > 0:
                            values.append(bin_data.median())

                    if values:
                        row[f"{metric}_median"] = np.median(values)
                        row[f"{metric}_mean"] = np.mean(values)
                        row[f"{metric}_std"] = np.std(values)
                        row[f"{metric}_p25"] = np.percentile(values, 25)
                        row[f"{metric}_p75"] = np.percentile(values, 75)
                        row[f"{metric}_n"] = len(values)

                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        if output_path is None:
            output_path = self.config.output.tables_path / "lifecycle_binned_metrics.csv"
        else:
            # Security: Sanitize user-provided path
            output_path = sanitize_output_path(output_path, self.config.output.tables_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported lifecycle metrics to {output_path}")
        return df

    def export_regime_transitions(
        self,
        dataset: MarketDataset,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Export regime transition matrix to CSV.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to export.
        output_path : Path | str | None
            Output file path.

        Returns
        -------
        pd.DataFrame
            Transition matrix DataFrame.
        """
        logger.info("Computing regime transition matrix")

        all_regimes = set()
        transition_counts = {}

        for contract in dataset:
            df = contract.data
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

        # Build matrix
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

        df = pd.DataFrame(matrix_norm, index=all_regimes, columns=all_regimes)
        df.index.name = "from_state"
        df = df.reset_index()

        if output_path is None:
            output_path = self.config.output.tables_path / "regime_transitions.csv"
        else:
            # Security: Sanitize user-provided path
            output_path = sanitize_output_path(output_path, self.config.output.tables_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported regime transitions to {output_path}")
        return df

    def export_all(
        self,
        dataset: MarketDataset,
        output_dir: Path | str | None = None,
    ) -> dict[str, Path]:
        """
        Export all metrics to CSV files.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to export.
        output_dir : Path | str | None
            Output directory.

        Returns
        -------
        dict[str, Path]
            Mapping from metric type to output path.
        """
        if output_dir is None:
            output_dir = self.config.output.tables_path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting all metrics...")

        paths = {}

        self.export_contract_summary(
            dataset,
            output_dir / "contract_summary_extended.csv",
        )
        paths["contract_summary"] = output_dir / "contract_summary_extended.csv"

        self.export_category_summary(
            dataset,
            output_dir / "category_summary_extended.csv",
        )
        paths["category_summary"] = output_dir / "category_summary_extended.csv"

        self.export_lifecycle_binned(
            dataset,
            n_bins=50,
            output_path=output_dir / "lifecycle_binned_metrics.csv",
        )
        paths["lifecycle_binned"] = output_dir / "lifecycle_binned_metrics.csv"

        self.export_regime_transitions(
            dataset,
            output_dir / "regime_transitions.csv",
        )
        paths["regime_transitions"] = output_dir / "regime_transitions.csv"

        logger.info(f"Exported {len(paths)} metric files to {output_dir}")
        return paths


def export_metrics(
    dataset: MarketDataset,
    config: IRPConfig | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Convenience function for exporting all metrics.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to export.
    config : IRPConfig | None
        Platform configuration.
    output_dir : Path | str | None
        Output directory.

    Returns
    -------
    dict[str, Path]
        Mapping from metric type to output path.
    """
    exporter = MetricsExporter(config)
    return exporter.export_all(dataset, output_dir)
