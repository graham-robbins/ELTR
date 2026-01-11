"""
Visualization module for ELTR platform.

Provides comprehensive plotting capabilities for prediction market
analysis including time series, heatmaps, trajectories, and
microstructure visualizations.

Research plot types include:
- Spread Tightening Curves (normalized time vs normalized spread)
- Liquidity Trajectories (lifecycle ratio vs normalized volume/depth)
- Volatility Burst Maps (heatmaps by lifecycle bin)
- Regime Occupancy Diagrams (bar charts of regime proportions)
- Category Fingerprints (radar charts comparing categories)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import ELTRConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    Category,
    ContractID,
    ContractTimeseries,
    MarketDataset,
)
from src.utils.normalization import normalize_series, normalize_dataframe
from src.utils.time_binning import (
    compute_lifecycle_features,
    make_lifecycle_bins,
    aggregate_by_bins,
    compute_binned_trajectory,
)

logger = get_logger("plots")


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    def __init__(self, config: ELTRConfig | None = None):
        self.config = config or get_config()
        self.plot_config = self.config.plots
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style."""
        try:
            plt.style.use(self.plot_config.style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

        plt.rcParams["figure.dpi"] = self.plot_config.dpi
        plt.rcParams["savefig.dpi"] = self.plot_config.dpi
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10

    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """Generate plot."""
        pass

    def save(self, fig: plt.Figure, filename: str, path: Path | None = None) -> Path:
        """Save figure to file."""
        if path is None:
            path = self.config.output.figures_path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        filepath = path / f"{filename}.{self.plot_config.save_format}"
        fig.savefig(
            filepath,
            format=self.plot_config.save_format,
            transparent=self.plot_config.transparent,
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.info(f"Saved figure: {filepath}")
        return filepath


class TimeSeriesPlotter(BasePlotter):
    """
    Plots time series data for contracts.

    Generates multi-panel charts showing price, volume,
    spread, and volatility evolution.
    """

    def plot(
        self,
        contract: ContractTimeseries,
        show_volume: bool = True,
        show_spread: bool = True,
        show_volatility: bool = True,
    ) -> plt.Figure:
        """
        Generate 4-panel time series chart.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to plot.
        show_volume : bool
            Include volume panel.
        show_spread : bool
            Include spread panel.
        show_volatility : bool
            Include volatility panel.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        df = contract.data
        n_panels = 1 + sum([show_volume, show_spread, show_volatility])

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=self.plot_config.figsize.quad,
            sharex=True,
        )

        if n_panels == 1:
            axes = [axes]

        ax_idx = 0

        self._plot_price(axes[ax_idx], df, contract.contract_id)
        ax_idx += 1

        if show_volume and "volume" in df.columns:
            self._plot_volume(axes[ax_idx], df)
            ax_idx += 1

        if show_spread and "spread" in df.columns:
            self._plot_spread(axes[ax_idx], df)
            ax_idx += 1

        if show_volatility and "volatility_short" in df.columns:
            self._plot_volatility(axes[ax_idx], df)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        fig.suptitle(
            f"{contract.contract_id} ({contract.category})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        return fig

    def _plot_price(self, ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
        """Plot price panel."""
        price_col = "price_c" if "price_c" in df.columns else "price"

        if price_col in df.columns:
            ax.plot(df.index, df[price_col], color="#1f77b4", linewidth=1)

        if "yes_bid_c" in df.columns and "yes_ask_c" in df.columns:
            ax.fill_between(
                df.index,
                df["yes_bid_c"],
                df["yes_ask_c"],
                alpha=0.2,
                color="#1f77b4",
                label="Bid-Ask",
            )

        ax.set_ylabel("Price (cents)")
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_title("Price & Bid-Ask Spread")

    def _plot_volume(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot volume panel."""
        ax.bar(df.index, df["volume"], width=0.0005, color="#2ca02c", alpha=0.7)

        if "volume_ma" in df.columns:
            ax.plot(
                df.index, df["volume_ma"],
                color="#d62728", linewidth=1,
                label="20-period MA",
            )
            ax.legend(loc="upper right")

        ax.set_ylabel("Volume")
        ax.set_title("Trading Volume")

    def _plot_spread(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot spread panel."""
        ax.plot(df.index, df["spread"], color="#ff7f0e", linewidth=1)

        if "spread_ma" in df.columns:
            ax.plot(
                df.index, df["spread_ma"],
                color="#9467bd", linewidth=1,
                linestyle="--",
                label="MA",
            )
            ax.legend(loc="upper right")

        ax.set_ylabel("Spread (cents)")
        ax.set_title("Bid-Ask Spread")

    def _plot_volatility(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Plot volatility panel."""
        if "volatility_short" in df.columns:
            ax.plot(
                df.index, df["volatility_short"],
                color="#d62728", linewidth=1,
                label="Short",
            )

        if "volatility_medium" in df.columns:
            ax.plot(
                df.index, df["volatility_medium"],
                color="#ff7f0e", linewidth=1,
                label="Medium",
            )

        if "volatility_long" in df.columns:
            ax.plot(
                df.index, df["volatility_long"],
                color="#2ca02c", linewidth=1,
                label="Long",
            )

        ax.set_ylabel("Volatility")
        ax.set_title("Rolling Volatility")
        ax.legend(loc="upper right")


class HeatmapPlotter(BasePlotter):
    """
    Generates heatmap visualizations.

    Creates liquidity heatmaps, correlation matrices,
    and category comparison charts.
    """

    def plot(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        cmap: str | None = None,
        annot: bool = True,
        fmt: str = ".2f",
    ) -> plt.Figure:
        """
        Generate heatmap.

        Parameters
        ----------
        data : pd.DataFrame
            Data matrix to visualize.
        title : str
            Plot title.
        cmap : str | None
            Colormap name.
        annot : bool
            Show annotations.
        fmt : str
            Annotation format.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=self.plot_config.figsize.heatmap)

        if cmap is None:
            cmap = self.plot_config.colormap

        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            center=0,
            square=True,
            linewidths=0.5,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        return fig

    def plot_liquidity_heatmap(
        self,
        dataset: MarketDataset,
        metric: str = "volume",
        resample_freq: str = "h",
    ) -> plt.Figure:
        """
        Generate liquidity heatmap across contracts.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        metric : str
            Metric for heatmap values.
        resample_freq : str
            Resampling frequency.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        data_dict = {}
        for contract in dataset:
            if metric in contract.data.columns:
                resampled = contract.data[metric].resample(resample_freq).sum()
                data_dict[contract.contract_id[:20]] = resampled

        if not data_dict:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        df = pd.DataFrame(data_dict).T
        df = df.iloc[:, :48]  # Limit columns for readability

        return self.plot(df, title=f"Liquidity Heatmap ({metric})", annot=False)


class CategoryPlotter(BasePlotter):
    """
    Generates category-level visualizations.

    Creates comparison charts and stacked plots
    across market categories.
    """

    def plot(
        self,
        dataset: MarketDataset,
        metric: str = "volume",
        plot_type: str = "bar",
    ) -> plt.Figure:
        """
        Generate category comparison plot.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        metric : str
            Metric to compare.
        plot_type : str
            Type of plot (bar, stacked, line).

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        category_data = self._aggregate_by_category(dataset, metric)

        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        colors = [
            self.plot_config.category_colors.get(cat, "#333333")
            for cat in category_data.keys()
        ]

        if plot_type == "bar":
            bars = ax.bar(
                category_data.keys(),
                category_data.values(),
                color=colors,
            )
            ax.bar_label(bars, fmt="%.0f")

        elif plot_type == "pie":
            ax.pie(
                category_data.values(),
                labels=category_data.keys(),
                colors=colors,
                autopct="%1.1f%%",
            )

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Category")
        fig.tight_layout()

        return fig

    def plot_stacked_timeseries(
        self,
        dataset: MarketDataset,
        metric: str = "volume",
        resample_freq: str = "D",
    ) -> plt.Figure:
        """
        Generate stacked area chart by category.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        metric : str
            Metric to stack.
        resample_freq : str
            Resampling frequency.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        category_series = {}

        for category in dataset.categories:
            cat_data = dataset.filter_by_category(category)
            combined = []

            for contract in cat_data:
                if metric in contract.data.columns:
                    combined.append(contract.data[metric])

            if combined:
                merged = pd.concat(combined, axis=0)
                resampled = merged.resample(resample_freq).sum()
                category_series[category] = resampled

        if not category_series:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        df = pd.DataFrame(category_series).fillna(0)

        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        colors = [
            self.plot_config.category_colors.get(cat, "#333333")
            for cat in df.columns
        ]

        ax.stackplot(df.index, df.T.values, labels=df.columns, colors=colors, alpha=0.8)
        ax.legend(loc="upper left")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Stacked {metric.replace('_', ' ').title()} by Category")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        fig.tight_layout()

        return fig

    def _aggregate_by_category(
        self, dataset: MarketDataset, metric: str
    ) -> dict[str, float]:
        """Aggregate metric by category."""
        result = {}
        for category in dataset.categories:
            cat_data = dataset.filter_by_category(category)
            total = 0
            for contract in cat_data:
                if metric in contract.data.columns:
                    total += contract.data[metric].sum()
            result[category] = total
        return result


class TrajectoryPlotter(BasePlotter):
    """
    Plots event-aligned trajectories.

    Creates normalized trajectory charts with
    confidence bands.
    """

    def plot(
        self,
        trajectories: list[pd.DataFrame],
        title: str = "Normalized Trajectory",
        show_individual: bool = False,
    ) -> plt.Figure:
        """
        Plot normalized trajectories.

        Parameters
        ----------
        trajectories : list[pd.DataFrame]
            List of trajectory DataFrames.
        title : str
            Plot title.
        show_individual : bool
            Show individual trajectories.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        if show_individual:
            for i, traj in enumerate(trajectories):
                if "normalized" in traj.columns:
                    ax.plot(
                        traj.index, traj["normalized"],
                        alpha=0.3, linewidth=0.5, color="#1f77b4",
                    )

        median_traj = self._compute_median(trajectories)
        if not median_traj.empty:
            ax.plot(
                median_traj.index, median_traj["median"],
                color="#d62728", linewidth=2, label="Median",
            )

            if "q25" in median_traj.columns and "q75" in median_traj.columns:
                ax.fill_between(
                    median_traj.index,
                    median_traj["q25"],
                    median_traj["q75"],
                    alpha=0.3,
                    color="#d62728",
                    label="IQR",
                )

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Event")
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Minutes to Event")
        ax.set_ylabel("Normalized Value")
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()

        return fig

    def _compute_median(self, trajectories: list[pd.DataFrame]) -> pd.DataFrame:
        """Compute median trajectory with confidence bands."""
        if not trajectories:
            return pd.DataFrame()

        all_data = []
        for traj in trajectories:
            if "normalized" in traj.columns:
                all_data.append(traj["normalized"].rename(len(all_data)))

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, axis=1)

        result = pd.DataFrame(index=combined.index)
        result["median"] = combined.median(axis=1)
        result["q25"] = combined.quantile(0.25, axis=1)
        result["q75"] = combined.quantile(0.75, axis=1)

        return result


class SpreadEvolutionPlotter(BasePlotter):
    """
    Plots spread evolution and dynamics.

    Creates spread tightening curves and
    regime-based spread analysis.
    """

    def plot(
        self,
        contract: ContractTimeseries,
        rolling_window: int = 20,
    ) -> plt.Figure:
        """
        Plot spread evolution.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to plot.
        rolling_window : int
            Rolling window for smoothing.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        df = contract.data

        if "spread" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No spread data available", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 1, figsize=self.plot_config.figsize.quad, sharex=True)

        axes[0].plot(df.index, df["spread"], alpha=0.5, linewidth=0.5, label="Raw")
        spread_ma = df["spread"].rolling(window=rolling_window).mean()
        axes[0].plot(df.index, spread_ma, color="#d62728", linewidth=1.5, label="MA")
        axes[0].set_ylabel("Spread (cents)")
        axes[0].set_title("Spread Evolution")
        axes[0].legend()

        if "spread_pct" in df.columns:
            axes[1].plot(df.index, df["spread_pct"] * 100, alpha=0.5, linewidth=0.5)
            spread_pct_ma = df["spread_pct"].rolling(window=rolling_window).mean() * 100
            axes[1].plot(df.index, spread_pct_ma, color="#d62728", linewidth=1.5)
            axes[1].set_ylabel("Spread (%)")
            axes[1].set_title("Relative Spread")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        fig.suptitle(f"Spread Analysis: {contract.contract_id}", fontweight="bold")
        fig.tight_layout()

        return fig


class DepthPlotter(BasePlotter):
    """
    Plots order book depth evolution.

    Creates depth visualization and resilience metrics.
    """

    def plot(
        self,
        contract: ContractTimeseries,
    ) -> plt.Figure:
        """
        Plot order book depth evolution.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to plot.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        df = contract.data

        fig, axes = plt.subplots(2, 1, figsize=self.plot_config.figsize.quad, sharex=True)

        if "yes_bid_c" in df.columns and "yes_ask_c" in df.columns:
            axes[0].fill_between(
                df.index,
                df["yes_bid_c"],
                df["yes_ask_c"],
                alpha=0.5,
                label="Bid-Ask Range",
            )
            axes[0].plot(df.index, df["yes_bid_c"], color="#2ca02c", linewidth=0.5, label="Bid")
            axes[0].plot(df.index, df["yes_ask_c"], color="#d62728", linewidth=0.5, label="Ask")
            axes[0].set_ylabel("Price (cents)")
            axes[0].set_title("Order Book Depth")
            axes[0].legend()

        if "depth_resilience" in df.columns:
            axes[1].plot(df.index, df["depth_resilience"], color="#9467bd", linewidth=1)
            axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            axes[1].set_ylabel("Resilience Score")
            axes[1].set_title("Depth Resilience")
        elif "book_thinning" in df.columns:
            axes[1].plot(df.index, df["book_thinning"], color="#ff7f0e", linewidth=1)
            axes[1].set_ylabel("Book Thinning")
            axes[1].set_title("Order Book Thinning")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        fig.suptitle(f"Depth Analysis: {contract.contract_id}", fontweight="bold")
        fig.tight_layout()

        return fig


class SpreadTighteningPlotter(BasePlotter):
    """
    Plots spread tightening curves on normalized time axis.

    Visualizes how bid-ask spreads evolve across the contract
    lifecycle with cross-contract comparison capability.
    """

    def plot(
        self,
        dataset: MarketDataset,
        n_bins: int = 50,
        show_individual: bool = False,
        by_category: bool = True,
    ) -> plt.Figure:
        """
        Generate spread tightening curve.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        n_bins : int
            Number of lifecycle bins.
        show_individual : bool
            Show individual contract curves.
        by_category : bool
            Separate curves by category.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        if by_category:
            categories = dataset.categories
        else:
            categories = [None]

        for category in categories:
            if category is not None:
                cat_data = dataset.filter_by_category(category)
                color = self.plot_config.category_colors.get(category, "#333333")
                label = category
            else:
                cat_data = dataset
                color = "#1f77b4"
                label = "All"

            all_curves = []
            for contract in cat_data:
                df = contract.data.copy()
                if "spread" not in df.columns:
                    continue

                if "lifecycle_ratio" not in df.columns:
                    df = compute_lifecycle_features(
                        df,
                        start_time=df.index.min(),
                        end_time=df.index.max(),
                    )

                df["lifecycle_bin"] = make_lifecycle_bins(df, n_bins=n_bins)
                binned = aggregate_by_bins(df, "lifecycle_bin", {"spread": "median"})

                if len(binned) > 5:
                    normalized = normalize_series(binned["spread_median"], method="minmax")
                    all_curves.append(normalized)

                    if show_individual:
                        ax.plot(
                            binned.index / n_bins,
                            normalized,
                            alpha=0.15, linewidth=0.5, color=color,
                        )

            if all_curves:
                combined = pd.concat(all_curves, axis=1)
                median_curve = combined.median(axis=1)
                q25 = combined.quantile(0.25, axis=1)
                q75 = combined.quantile(0.75, axis=1)

                x_axis = median_curve.index / n_bins
                ax.plot(x_axis, median_curve, color=color, linewidth=2, label=label)
                ax.fill_between(x_axis, q25, q75, alpha=0.2, color=color)

        ax.set_xlabel("Lifecycle Ratio (0 = Start, 1 = Settlement)")
        ax.set_ylabel("Normalized Spread")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.5, label="Resolution Zone")
        ax.legend(loc="upper right")
        ax.set_title("Spread Tightening Curve")
        fig.tight_layout()

        return fig


class LiquidityTrajectoryPlotter(BasePlotter):
    """
    Plots liquidity trajectories across contract lifecycle.

    Visualizes volume and depth dynamics on normalized axes
    for cross-contract comparison.
    """

    def plot(
        self,
        dataset: MarketDataset,
        metric: str = "volume",
        n_bins: int = 50,
        by_category: bool = True,
    ) -> plt.Figure:
        """
        Generate liquidity trajectory plot.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        metric : str
            Liquidity metric (volume, volume_zscore, depth_resilience).
        n_bins : int
            Number of lifecycle bins.
        by_category : bool
            Separate curves by category.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, axes = plt.subplots(2, 1, figsize=self.plot_config.figsize.quad, sharex=True)

        if by_category:
            categories = dataset.categories
        else:
            categories = [None]

        for category in categories:
            if category is not None:
                cat_data = dataset.filter_by_category(category)
                color = self.plot_config.category_colors.get(category, "#333333")
                label = category
            else:
                cat_data = dataset
                color = "#1f77b4"
                label = "All"

            volume_curves = []
            depth_curves = []

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

                vol_binned = aggregate_by_bins(df, "lifecycle_bin", {metric: "median"})
                if len(vol_binned) > 5:
                    normalized = normalize_series(vol_binned[f"{metric}_median"], method="zscore")
                    volume_curves.append(normalized)

                if "depth_resilience" in df.columns:
                    depth_binned = aggregate_by_bins(df, "lifecycle_bin", {"depth_resilience": "median"})
                    if len(depth_binned) > 5:
                        depth_curves.append(depth_binned["depth_resilience_median"])

            if volume_curves:
                combined = pd.concat(volume_curves, axis=1)
                median_curve = combined.median(axis=1)
                q25 = combined.quantile(0.25, axis=1)
                q75 = combined.quantile(0.75, axis=1)

                x_axis = median_curve.index / n_bins
                axes[0].plot(x_axis, median_curve, color=color, linewidth=2, label=label)
                axes[0].fill_between(x_axis, q25, q75, alpha=0.2, color=color)

            if depth_curves:
                combined = pd.concat(depth_curves, axis=1)
                median_curve = combined.median(axis=1)
                x_axis = median_curve.index / n_bins
                axes[1].plot(x_axis, median_curve, color=color, linewidth=2, label=label)

        axes[0].set_ylabel(f"Normalized {metric.replace('_', ' ').title()}")
        axes[0].set_title("Volume Trajectory")
        axes[0].legend(loc="upper right")
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        axes[1].set_xlabel("Lifecycle Ratio")
        axes[1].set_ylabel("Depth Resilience")
        axes[1].set_title("Depth Trajectory")
        axes[1].legend(loc="upper right")

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle("Liquidity Trajectory Analysis", fontweight="bold")
        fig.tight_layout()

        return fig


class VolatilityBurstPlotter(BasePlotter):
    """
    Creates dual volatility burst heatmaps by lifecycle bin (Section 12).

    Produces two separate heatmaps:
    - Mean Burst Intensity: Average volatility magnitude during burst events
    - Burst Frequency: Count/proportion of burst events per bin

    This separation prevents confounding intensity with frequency.
    """

    def plot(
        self,
        dataset: MarketDataset,
        n_bins: int = 20,
        volatility_metric: str = "volatility_short",
    ) -> plt.Figure:
        """
        Generate dual volatility burst heatmaps (Section 12).

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        n_bins : int
            Number of lifecycle bins.
        volatility_metric : str
            Volatility column to use.

        Returns
        -------
        plt.Figure
            Generated figure with two heatmaps:
            - Top: Mean burst intensity (average volatility during bursts)
            - Bottom: Burst frequency (proportion of observations that are bursts)
        """
        categories = dataset.categories
        n_categories = len(categories)

        if n_categories == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Section 12: Track burst intensity sum and count separately
        burst_intensity_sum = np.zeros((n_categories, n_bins))
        burst_counts = np.zeros((n_categories, n_bins))
        total_obs_counts = np.zeros((n_categories, n_bins))

        for cat_idx, category in enumerate(categories):
            cat_data = dataset.filter_by_category(category)

            for contract in cat_data:
                df = contract.data.copy()
                if volatility_metric not in df.columns:
                    continue

                if "lifecycle_ratio" not in df.columns:
                    df = compute_lifecycle_features(
                        df,
                        start_time=df.index.min(),
                        end_time=df.index.max(),
                    )

                df["lifecycle_bin"] = make_lifecycle_bins(df, n_bins=n_bins)

                vol_mean = df[volatility_metric].mean()
                vol_std = df[volatility_metric].std()
                if vol_std > 0:
                    burst_threshold = vol_mean + 2 * vol_std
                    df["is_burst"] = df[volatility_metric] > burst_threshold

                    for bin_idx in range(n_bins):
                        bin_data = df[df["lifecycle_bin"] == bin_idx]
                        if len(bin_data) > 0:
                            total_obs_counts[cat_idx, bin_idx] += len(bin_data)

                            # Only sum volatility from burst observations
                            burst_data = bin_data[bin_data["is_burst"]]
                            if len(burst_data) > 0:
                                burst_intensity_sum[cat_idx, bin_idx] += burst_data[volatility_metric].sum()
                                burst_counts[cat_idx, bin_idx] += len(burst_data)

        with np.errstate(all="ignore"):
            # Section 12: Mean burst intensity = sum of burst volatilities / number of bursts
            mean_burst_intensity = np.where(
                burst_counts > 0,
                burst_intensity_sum / burst_counts,
                0.0,
            )

            # Section 12: Burst frequency = proportion of observations that are bursts
            burst_frequency = np.where(
                total_obs_counts > 0,
                burst_counts / total_obs_counts,
                0.0,
            )

        fig, axes = plt.subplots(2, 1, figsize=self.plot_config.figsize.quad)

        # Top: Mean Burst Intensity (Section 12)
        im1 = axes[0].imshow(
            mean_burst_intensity, aspect="auto", cmap="YlOrRd",
            extent=[0, 1, n_categories - 0.5, -0.5],
        )
        axes[0].set_yticks(range(n_categories))
        axes[0].set_yticklabels(categories)
        axes[0].set_xlabel("Lifecycle Ratio")
        axes[0].set_title("Mean Burst Intensity (Avg Volatility During Bursts)")
        plt.colorbar(im1, ax=axes[0], label="Mean Intensity")

        # Bottom: Burst Frequency (Section 12)
        im2 = axes[1].imshow(
            burst_frequency, aspect="auto", cmap="Blues",
            extent=[0, 1, n_categories - 0.5, -0.5],
        )
        axes[1].set_yticks(range(n_categories))
        axes[1].set_yticklabels(categories)
        axes[1].set_xlabel("Lifecycle Ratio")
        axes[1].set_title("Burst Frequency (Proportion of Burst Events)")
        plt.colorbar(im2, ax=axes[1], label="Frequency")

        fig.suptitle("Dual Volatility Burst Heatmaps (Section 12)", fontweight="bold")
        fig.tight_layout()

        return fig


class RegimeOccupancyPlotter(BasePlotter):
    """
    Plots regime occupancy diagrams.

    Visualizes the proportion of time spent in each
    microstructure regime by category.
    """

    REGIME_COLORS = {
        "FROZEN": "#9e9e9e",
        "THIN": "#ffeb3b",
        "NORMAL": "#4caf50",
        "ACTIVE_INFORMATION": "#2196f3",
        "VOLATILITY_BURST": "#f44336",
        "RESOLUTION_DRIFT": "#9c27b0",
        "UNKNOWN": "#607d8b",
    }

    def plot(
        self,
        dataset: MarketDataset,
        by_category: bool = True,
        stacked: bool = True,
    ) -> plt.Figure:
        """
        Generate regime occupancy diagram.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        by_category : bool
            Separate by category.
        stacked : bool
            Use stacked bar chart.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        if by_category:
            categories = dataset.categories
        else:
            categories = ["All"]

        regime_counts = {cat: {} for cat in categories}

        for contract in dataset:
            df = contract.data
            if "microstructure_state" not in df.columns:
                continue

            cat = contract.category if by_category else "All"
            state_counts = df["microstructure_state"].value_counts()

            for state, count in state_counts.items():
                state_name = str(state).replace("MicrostructureState.", "")
                if state_name not in regime_counts[cat]:
                    regime_counts[cat][state_name] = 0
                regime_counts[cat][state_name] += count

        all_regimes = set()
        for cat_regimes in regime_counts.values():
            all_regimes.update(cat_regimes.keys())

        if not all_regimes:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No regime data available", ha="center", va="center")
            return fig

        all_regimes = sorted(all_regimes)
        regime_data = {regime: [] for regime in all_regimes}

        for cat in categories:
            cat_total = sum(regime_counts[cat].values()) or 1
            for regime in all_regimes:
                count = regime_counts[cat].get(regime, 0)
                regime_data[regime].append(count / cat_total * 100)

        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        x = np.arange(len(categories))
        width = 0.7

        if stacked:
            bottom = np.zeros(len(categories))
            for regime in all_regimes:
                color = self.REGIME_COLORS.get(regime, "#333333")
                ax.bar(
                    x, regime_data[regime],
                    width, label=regime, bottom=bottom,
                    color=color,
                )
                bottom += np.array(regime_data[regime])
        else:
            bar_width = width / len(all_regimes)
            for i, regime in enumerate(all_regimes):
                color = self.REGIME_COLORS.get(regime, "#333333")
                offset = (i - len(all_regimes) / 2 + 0.5) * bar_width
                ax.bar(
                    x + offset, regime_data[regime],
                    bar_width, label=regime, color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("Regime Occupancy (%)")
        ax.set_title("Microstructure Regime Occupancy by Category")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax.set_ylim(0, 105)

        fig.tight_layout()

        return fig

    def plot_transitions(
        self,
        dataset: MarketDataset,
    ) -> plt.Figure:
        """
        Plot regime transition matrix.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        all_regimes = set()
        transition_counts = {}

        for contract in dataset:
            df = contract.data
            if "microstructure_state" not in df.columns:
                continue

            states = df["microstructure_state"].astype(str).str.replace("MicrostructureState.", "")
            all_regimes.update(states.unique())

            for i in range(len(states) - 1):
                from_state = states.iloc[i]
                to_state = states.iloc[i + 1]
                key = (from_state, to_state)
                transition_counts[key] = transition_counts.get(key, 0) + 1

        if not all_regimes:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No regime data available", ha="center", va="center")
            return fig

        all_regimes = sorted(all_regimes)
        n_regimes = len(all_regimes)
        matrix = np.zeros((n_regimes, n_regimes))

        for (from_state, to_state), count in transition_counts.items():
            if from_state in all_regimes and to_state in all_regimes:
                i = all_regimes.index(from_state)
                j = all_regimes.index(to_state)
                matrix[i, j] = count

        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            matrix_norm = np.divide(matrix, row_sums, where=row_sums > 0)
            matrix_norm = np.nan_to_num(matrix_norm)

        fig, ax = plt.subplots(figsize=self.plot_config.figsize.heatmap)

        sns.heatmap(
            matrix_norm, ax=ax,
            xticklabels=all_regimes,
            yticklabels=all_regimes,
            annot=True, fmt=".2f",
            cmap="Blues",
            vmin=0, vmax=1,
        )

        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        ax.set_title("Regime Transition Probabilities")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        fig.tight_layout()

        return fig


class CategoryFingerprintPlotter(BasePlotter):
    """
    Creates category fingerprint radar charts.

    Compares categories across multiple normalized
    microstructure metrics.
    """

    DEFAULT_METRICS = [
        "spread", "volume", "volatility_short",
        "depth_resilience", "book_thinning",
    ]

    def plot(
        self,
        dataset: MarketDataset,
        metrics: list[str] | None = None,
    ) -> plt.Figure:
        """
        Generate category fingerprint radar chart.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        metrics : list[str] | None
            Metrics to include in fingerprint.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        if metrics is None:
            metrics = self.DEFAULT_METRICS

        categories = dataset.categories
        if len(categories) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        category_profiles = {cat: {} for cat in categories}

        for category in categories:
            cat_data = dataset.filter_by_category(category)
            for metric in metrics:
                values = []
                for contract in cat_data:
                    if metric in contract.data.columns:
                        values.append(contract.data[metric].median())
                if values:
                    category_profiles[category][metric] = np.median(values)
                else:
                    category_profiles[category][metric] = 0

        available_metrics = [m for m in metrics if any(
            m in profile for profile in category_profiles.values()
        )]

        if len(available_metrics) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient metrics for radar chart", ha="center", va="center")
            return fig

        all_values = {m: [] for m in available_metrics}
        for profile in category_profiles.values():
            for m in available_metrics:
                all_values[m].append(profile.get(m, 0))

        normalized_profiles = {cat: {} for cat in categories}
        for m in available_metrics:
            min_val = min(all_values[m])
            max_val = max(all_values[m])
            range_val = max_val - min_val if max_val > min_val else 1
            for cat in categories:
                val = category_profiles[cat].get(m, 0)
                normalized_profiles[cat][m] = (val - min_val) / range_val

        n_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single, subplot_kw=dict(polar=True))

        for category in categories:
            values = [normalized_profiles[category].get(m, 0) for m in available_metrics]
            values += values[:1]
            color = self.plot_config.category_colors.get(category, "#333333")
            ax.plot(angles, values, linewidth=2, label=category, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in available_metrics])
        ax.set_ylim(0, 1.1)

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        ax.set_title("Category Microstructure Fingerprints", pad=20)

        fig.tight_layout()

        return fig


class EventTrajectoryPlotter(BasePlotter):
    """
    Plots event-aligned trajectories from event alignment module.

    Creates signature trajectory plots with median curves
    and percentile bands.
    """

    def plot(
        self,
        aggregated_trajectory,
        title: str | None = None,
        show_bands: bool = True,
    ) -> plt.Figure:
        """
        Plot aggregated trajectory.

        Parameters
        ----------
        aggregated_trajectory : AggregatedTrajectory
            Aggregated trajectory from event alignment.
        title : str | None
            Plot title.
        show_bands : bool
            Show percentile bands.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=self.plot_config.figsize.single)

        time_hours = aggregated_trajectory.time_grid / 60

        ax.plot(
            time_hours, aggregated_trajectory.median,
            color="#d62728", linewidth=2, label="Median",
        )

        if show_bands:
            ax.fill_between(
                time_hours,
                aggregated_trajectory.p25,
                aggregated_trajectory.p75,
                alpha=0.3, color="#d62728", label="IQR (25-75)",
            )
            ax.fill_between(
                time_hours,
                aggregated_trajectory.p10,
                aggregated_trajectory.p90,
                alpha=0.15, color="#d62728", label="90% Band",
            )

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7, label="Event")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Hours Relative to Event")
        ax.set_ylabel(f"Normalized {aggregated_trajectory.metric.replace('_', ' ').title()}")

        if title is None:
            cat_str = f" ({aggregated_trajectory.category})" if aggregated_trajectory.category else ""
            title = f"{aggregated_trajectory.metric.replace('_', ' ').title()} Trajectory{cat_str}"

        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()

        return fig

    def plot_multi_metric(
        self,
        trajectories: dict[str, Any],
        metrics: list[str] | None = None,
    ) -> plt.Figure:
        """
        Plot multiple metrics in subplots.

        Parameters
        ----------
        trajectories : dict[str, list[AggregatedTrajectory]]
            Trajectories from EventTrajectoryAnalyzer.
        metrics : list[str] | None
            Metrics to plot.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        if metrics is None:
            metrics = list(trajectories.keys())[:6]

        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.plot_config.figsize.single[0] * n_cols,
                     self.plot_config.figsize.single[1] * n_rows),
            squeeze=False,
        )

        for idx, metric in enumerate(metrics):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            if metric not in trajectories or not trajectories[metric]:
                ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
                continue

            overall = [t for t in trajectories[metric] if t.category is None]
            if overall:
                traj = overall[0]
                time_hours = traj.time_grid / 60

                ax.plot(time_hours, traj.median, color="#d62728", linewidth=2)
                ax.fill_between(
                    time_hours, traj.p25, traj.p75,
                    alpha=0.3, color="#d62728",
                )

            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Hours to Event")

        for idx in range(n_metrics, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle("Event-Aligned Signature Trajectories", fontweight="bold", y=1.02)
        fig.tight_layout()

        return fig


class PlotManager:
    """
    Main plotting orchestrator.

    Coordinates all plotting operations and provides
    unified interface for visualization generation.

    Includes new research plotters:
    - spread_tightening: Lifecycle-normalized spread curves
    - liquidity_trajectory: Volume/depth over lifecycle
    - volatility_burst: Heatmaps of volatility by lifecycle
    - regime_occupancy: Regime proportions by category
    - fingerprint: Radar charts comparing categories
    - event_trajectory: Event-aligned signature plots
    """

    def __init__(self, config: ELTRConfig | None = None):
        """
        Initialize plot manager.

        Parameters
        ----------
        config : ELTRConfig | None
            Platform configuration.
        """
        self.config = config or get_config()

        # Original plotters
        self.timeseries = TimeSeriesPlotter(config)
        self.heatmap = HeatmapPlotter(config)
        self.category = CategoryPlotter(config)
        self.trajectory = TrajectoryPlotter(config)
        self.spread = SpreadEvolutionPlotter(config)
        self.depth = DepthPlotter(config)

        # New research plotters
        self.spread_tightening = SpreadTighteningPlotter(config)
        self.liquidity_trajectory = LiquidityTrajectoryPlotter(config)
        self.volatility_burst = VolatilityBurstPlotter(config)
        self.regime_occupancy = RegimeOccupancyPlotter(config)
        self.fingerprint = CategoryFingerprintPlotter(config)
        self.event_trajectory = EventTrajectoryPlotter(config)

    def generate_contract_report(
        self,
        contract: ContractTimeseries,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Generate all plots for a single contract.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to visualize.
        output_dir : Path | None
            Output directory.

        Returns
        -------
        list[Path]
            Paths to saved figures.
        """
        if output_dir is None:
            output_dir = self.config.output.figures_path

        output_dir = Path(output_dir)
        saved = []

        fig = self.timeseries.plot(contract)
        path = self.timeseries.save(fig, f"{contract.contract_id}_timeseries", output_dir)
        saved.append(path)

        fig = self.spread.plot(contract)
        path = self.spread.save(fig, f"{contract.contract_id}_spread", output_dir)
        saved.append(path)

        fig = self.depth.plot(contract)
        path = self.depth.save(fig, f"{contract.contract_id}_depth", output_dir)
        saved.append(path)

        return saved

    def generate_dataset_report(
        self,
        dataset: MarketDataset,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Generate all plots for entire dataset.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        output_dir : Path | None
            Output directory.

        Returns
        -------
        list[Path]
            Paths to saved figures.
        """
        if output_dir is None:
            output_dir = self.config.output.figures_path

        output_dir = Path(output_dir)
        saved = []

        logger.info(f"Generating dataset plots for {len(dataset)} contracts")

        # Original plots
        fig = self.category.plot(dataset, metric="volume", plot_type="bar")
        path = self.category.save(fig, "category_volume_comparison", output_dir)
        saved.append(path)

        fig = self.category.plot_stacked_timeseries(dataset, metric="volume")
        path = self.category.save(fig, "stacked_volume", output_dir)
        saved.append(path)

        fig = self.heatmap.plot_liquidity_heatmap(dataset, metric="volume")
        path = self.heatmap.save(fig, "liquidity_heatmap", output_dir)
        saved.append(path)

        # New research plots
        logger.info("Generating research plots...")

        fig = self.spread_tightening.plot(dataset, n_bins=50, by_category=True)
        path = self.spread_tightening.save(fig, "spread_tightening_curve", output_dir)
        saved.append(path)

        fig = self.liquidity_trajectory.plot(dataset, metric="volume", n_bins=50)
        path = self.liquidity_trajectory.save(fig, "liquidity_trajectory", output_dir)
        saved.append(path)

        fig = self.volatility_burst.plot(dataset, n_bins=20)
        path = self.volatility_burst.save(fig, "volatility_burst_map", output_dir)
        saved.append(path)

        fig = self.regime_occupancy.plot(dataset, by_category=True, stacked=True)
        path = self.regime_occupancy.save(fig, "regime_occupancy", output_dir)
        saved.append(path)

        fig = self.regime_occupancy.plot_transitions(dataset)
        path = self.regime_occupancy.save(fig, "regime_transitions", output_dir)
        saved.append(path)

        fig = self.fingerprint.plot(dataset)
        path = self.fingerprint.save(fig, "category_fingerprints", output_dir)
        saved.append(path)

        logger.info(f"Generated {len(saved)} dataset figures")
        return saved

    def generate_microstructure_atlas(
        self,
        dataset: MarketDataset,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Generate complete microstructure atlas for publication.

        Creates all research-grade plots for ELTR:
        - Spread tightening curves
        - Liquidity trajectories
        - Volatility burst maps
        - Regime occupancy diagrams
        - Category fingerprints

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to visualize.
        output_dir : Path | None
            Output directory.

        Returns
        -------
        list[Path]
            Paths to saved figures.
        """
        if output_dir is None:
            output_dir = self.config.output.figures_path / "atlas"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        logger.info("Generating Microstructure Atlas...")

        # Spread tightening variants
        fig = self.spread_tightening.plot(dataset, n_bins=50, by_category=True)
        path = self.spread_tightening.save(fig, "atlas_spread_tightening_by_category", output_dir)
        saved.append(path)

        fig = self.spread_tightening.plot(dataset, n_bins=100, by_category=False, show_individual=True)
        path = self.spread_tightening.save(fig, "atlas_spread_tightening_all", output_dir)
        saved.append(path)

        # Liquidity trajectory variants
        fig = self.liquidity_trajectory.plot(dataset, metric="volume", n_bins=50)
        path = self.liquidity_trajectory.save(fig, "atlas_liquidity_volume", output_dir)
        saved.append(path)

        # Volatility burst maps
        fig = self.volatility_burst.plot(dataset, n_bins=20)
        path = self.volatility_burst.save(fig, "atlas_volatility_burst", output_dir)
        saved.append(path)

        # Regime analysis
        fig = self.regime_occupancy.plot(dataset, by_category=True, stacked=True)
        path = self.regime_occupancy.save(fig, "atlas_regime_occupancy_stacked", output_dir)
        saved.append(path)

        fig = self.regime_occupancy.plot(dataset, by_category=True, stacked=False)
        path = self.regime_occupancy.save(fig, "atlas_regime_occupancy_grouped", output_dir)
        saved.append(path)

        fig = self.regime_occupancy.plot_transitions(dataset)
        path = self.regime_occupancy.save(fig, "atlas_regime_transitions", output_dir)
        saved.append(path)

        # Category fingerprints
        fig = self.fingerprint.plot(dataset)
        path = self.fingerprint.save(fig, "atlas_category_fingerprints", output_dir)
        saved.append(path)

        logger.info(f"Generated Microstructure Atlas with {len(saved)} figures")
        return saved


def generate_all_plots(
    dataset: MarketDataset,
    config: ELTRConfig | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Convenience function to generate all visualizations.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to visualize.
    config : ELTRConfig | None
        Platform configuration.
    output_dir : Path | None
        Output directory.

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    manager = PlotManager(config)
    return manager.generate_dataset_report(dataset, output_dir)
