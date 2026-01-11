"""
ELTR: Episodic Liquidity and Trading Regimes in Prediction Markets

Production-ready analytical framework for prediction market microstructure analysis.
Implements a pipelines-first architecture: ingestion → cleaning →
normalization → feature extraction → analytics → visualization → export.

Usage:
    python ELTR.py --event-type sports --save-figures
    python ELTR.py --categories Sports,Politics --output-dir results
    python ELTR.py --full-analysis
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

np.random.seed(42)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from src.utils.config import ELTRConfig, load_config, get_config
from src.utils.logging import setup_logging, get_logger
from src.utils.types import MarketDataset, Category

from src.ingest.kalshi_loader import KalshiDataLoader, load_kalshi_data
from src.clean.cleaner import DataCleaner, clean_market_data, CleaningStats
from src.features.feature_engineering import FeatureEngineer, engineer_features
from src.microstructure.analysis import MicrostructureAnalyzer, analyze_microstructure
from src.microstructure.event_alignment import EventTrajectoryAnalyzer, compute_event_trajectories
from src.plots.plotting import PlotManager, generate_all_plots
from src.utils.export import MetricsExporter, export_metrics


class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str, config: ELTRConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"pipeline.{name}")

    def run(self, data: Any) -> Any:
        """Execute pipeline stage."""
        raise NotImplementedError


class IngestionStage(PipelineStage):
    """Data ingestion stage."""

    def __init__(self, config: ELTRConfig, categories: list[Category] | None = None):
        super().__init__("ingestion", config)
        self.categories = categories
        self.loader = KalshiDataLoader(config)

    def run(self, data: Any = None) -> MarketDataset:
        """Load all market data."""
        self.logger.info("Starting data ingestion")
        dataset = self.loader.load_all(
            categories=self.categories,
            min_observations=self.config.cleaning.min_observations,
        )
        self.logger.info(f"Ingested {len(dataset)} contracts")
        return dataset


class CleaningStage(PipelineStage):
    """Data cleaning stage."""

    def __init__(self, config: ELTRConfig):
        super().__init__("cleaning", config)
        self.cleaner = DataCleaner(config)

    def run(self, data: MarketDataset) -> tuple[MarketDataset, dict[str, CleaningStats]]:
        """Clean market data."""
        self.logger.info("Starting data cleaning")
        cleaned, stats = self.cleaner.clean_dataset(data)
        self.logger.info(f"Cleaned {len(cleaned)} contracts")
        return cleaned, stats


class FeatureStage(PipelineStage):
    """Feature engineering stage."""

    def __init__(self, config: ELTRConfig):
        super().__init__("features", config)
        self.engineer = FeatureEngineer(config)

    def run(self, data: MarketDataset) -> MarketDataset:
        """Engineer features for all contracts."""
        self.logger.info("Starting feature engineering")
        featured = self.engineer.engineer_dataset(data)
        self.logger.info(f"Engineered features for {len(featured)} contracts")
        return featured


class MicrostructureStage(PipelineStage):
    """Microstructure analysis stage."""

    def __init__(self, config: ELTRConfig):
        super().__init__("microstructure", config)
        self.analyzer = MicrostructureAnalyzer(config)

    def run(self, data: MarketDataset) -> tuple[MarketDataset, pd.DataFrame]:
        """Analyze market microstructure."""
        self.logger.info("Starting microstructure analysis")
        metrics, summary_df = self.analyzer.analyze_dataset(data)
        self.logger.info(f"Analyzed microstructure for {len(metrics)} contracts")
        return data, summary_df


class VisualizationStage(PipelineStage):
    """Visualization generation stage."""

    def __init__(self, config: ELTRConfig, output_dir: Path | None = None):
        super().__init__("visualization", config)
        self.output_dir = output_dir or config.output.figures_path
        self.plot_manager = PlotManager(config)

    def run(self, data: MarketDataset) -> list[Path]:
        """Generate all visualizations."""
        self.logger.info("Starting visualization generation")
        saved_paths = self.plot_manager.generate_dataset_report(data, self.output_dir)
        self.logger.info(f"Generated {len(saved_paths)} figures")
        return saved_paths


class ExportStage(PipelineStage):
    """Data export stage with extended metrics."""

    def __init__(self, config: ELTRConfig, output_dir: Path | None = None):
        super().__init__("export", config)
        self.output_dir = output_dir or Path(config.output.tables_path)
        self.metrics_exporter = MetricsExporter(config)

    def run(self, data: dict[str, Any]) -> dict[str, Path]:
        """Export analysis results with extended metrics."""
        self.logger.info("Starting data export")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exported = {}

        # Basic contract summary
        if "dataset" in data:
            summary_path = self.output_dir / "contract_summary.csv"
            data["dataset"].summary().to_csv(summary_path, index=False)
            exported["contract_summary"] = summary_path

            # Extended metrics export
            self.logger.info("Exporting extended metrics...")
            extended_paths = self.metrics_exporter.export_all(
                data["dataset"],
                self.output_dir,
            )
            exported.update(extended_paths)

        # Microstructure summary
        if "microstructure_summary" in data:
            micro_path = self.output_dir / "microstructure_summary.csv"
            data["microstructure_summary"].to_csv(micro_path, index=False)
            exported["microstructure_summary"] = micro_path

        # Cleaning stats
        if "cleaning_stats" in data:
            stats_records = []
            for contract_id, stats in data["cleaning_stats"].items():
                stats_records.append({
                    "contract_id": contract_id,
                    "original_rows": stats.original_rows,
                    "final_rows": stats.final_rows,
                    "rows_removed": stats.rows_removed,
                    "removal_pct": stats.removal_pct,
                    "nulls_imputed": stats.nulls_imputed,
                    "outliers_winsorized": stats.outliers_winsorized,
                })
            if stats_records:
                stats_path = self.output_dir / "cleaning_stats.csv"
                pd.DataFrame(stats_records).to_csv(stats_path, index=False)
                exported["cleaning_stats"] = stats_path

        self.logger.info(f"Exported {len(exported)} files")
        return exported


class EventTrajectoryStage(PipelineStage):
    """Event trajectory analysis stage."""

    def __init__(self, config: ELTRConfig):
        super().__init__("event_trajectories", config)
        self.analyzer = EventTrajectoryAnalyzer(config)

    def run(self, data: MarketDataset) -> dict[str, Any]:
        """Compute event-aligned trajectories."""
        self.logger.info("Starting event trajectory analysis")
        trajectories = self.analyzer.analyze_dataset(data)
        signatures = self.analyzer.compute_signature_trajectories(data)
        self.logger.info(f"Computed trajectories for {len(trajectories)} metrics")
        return {
            "trajectories": trajectories,
            "signatures": signatures,
            "time_grid_hours": self.analyzer.get_time_grid_hours(),
        }


class BatchedResearchPipeline:
    """
    Memory-safe batched research pipeline.

    Processes contracts in batches to prevent memory crashes.
    Ingestion and cleaning happen once, then heavy stages
    (features, microstructure, event_trajectories) run per batch.
    """

    def __init__(
        self,
        config: ELTRConfig | None = None,
        categories: list[Category] | None = None,
        output_dir: Path | None = None,
        batch_size: int = 30,
        max_contracts: int | None = None,
    ):
        """
        Initialize batched pipeline.

        Parameters
        ----------
        config : ELTRConfig | None
            Platform configuration.
        categories : list[Category] | None
            Categories to analyze.
        output_dir : Path | None
            Output directory for results.
        batch_size : int
            Number of contracts per batch.
        max_contracts : int | None
            Maximum contracts to process (for testing).
        """
        self.config = config or get_config()
        self.categories = categories
        self.output_dir = output_dir or Path(self.config.output.base_path)
        self.batch_size = batch_size
        self.max_contracts = max_contracts
        self.logger = get_logger("pipeline.batched")

        # Stage instances
        self.ingestion_stage = IngestionStage(self.config, categories)
        self.cleaning_stage = CleaningStage(self.config)
        self.feature_stage = FeatureStage(self.config)
        self.microstructure_stage = MicrostructureStage(self.config)
        self.event_trajectory_stage = EventTrajectoryStage(self.config)
        self.visualization_stage = VisualizationStage(self.config, self.output_dir / "figures")
        self.export_stage = ExportStage(self.config, self.output_dir / "tables")

        self.results: dict[str, Any] = {}

    def run(
        self,
        save_figures: bool = True,
        export_tables: bool = True,
    ) -> dict[str, Any]:
        """
        Execute batched pipeline.

        Parameters
        ----------
        save_figures : bool
            Generate and save figures.
        export_tables : bool
            Export summary tables.

        Returns
        -------
        dict[str, Any]
            Pipeline results.
        """
        self.logger.info("=" * 60)
        self.logger.info("KALSHI PREDICTION MARKET RESEARCH PIPELINE (BATCHED)")
        self.logger.info("=" * 60)

        start_time = datetime.now()
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing batch output files for fresh run
        self._clear_batch_outputs(tables_dir)

        # STAGE 1: Ingestion (once)
        self.logger.info("[INGESTION] Starting...")
        dataset = self.ingestion_stage.run()
        self.logger.info(f"[INGESTION] Complete - {len(dataset)} contracts")

        # STAGE 2: Cleaning (once)
        self.logger.info("[CLEANING] Starting...")
        cleaned_dataset, cleaning_stats = self.cleaning_stage.run(dataset)
        self.results["cleaning_stats"] = cleaning_stats
        self.logger.info(f"[CLEANING] Complete - {len(cleaned_dataset)} contracts")

        # Free memory from raw dataset
        del dataset
        gc.collect()

        # Get contract list
        contract_ids = list(cleaned_dataset.contracts.keys())

        # Apply max_contracts limit
        if self.max_contracts is not None:
            contract_ids = contract_ids[:self.max_contracts]
            self.logger.info(f"Limited to {len(contract_ids)} contracts (--max-contracts)")

        # Create batches
        n_contracts = len(contract_ids)
        batches = [
            contract_ids[i:i + self.batch_size]
            for i in range(0, n_contracts, self.batch_size)
        ]
        n_batches = len(batches)

        self.logger.info(f"Processing {n_contracts} contracts in {n_batches} batches (size={self.batch_size})")

        # Accumulators for aggregated results
        all_microstructure_summaries = []
        all_trajectory_results = []
        processed_dataset = MarketDataset(metadata=cleaned_dataset.metadata.copy())

        # STAGE 3-5: Process each batch
        for batch_idx, batch_contract_ids in enumerate(batches):
            batch_num = batch_idx + 1
            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Processing {len(batch_contract_ids)} contracts...")

            # Create batch dataset
            batch_dataset = MarketDataset(metadata=cleaned_dataset.metadata.copy())
            for contract_id in batch_contract_ids:
                if contract_id in cleaned_dataset.contracts:
                    batch_dataset.add(cleaned_dataset.contracts[contract_id])

            # Feature engineering
            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Feature engineering...")
            featured_batch = self.feature_stage.run(batch_dataset)

            # Microstructure analysis
            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Microstructure analysis...")
            _, micro_summary = self.microstructure_stage.run(featured_batch)
            all_microstructure_summaries.append(micro_summary)

            # Event trajectories
            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Event trajectory analysis...")
            traj_results = self.event_trajectory_stage.run(featured_batch)
            all_trajectory_results.append(traj_results)

            # Export batch metrics (append mode)
            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Exporting batch metrics...")
            self._export_batch_metrics(featured_batch, tables_dir, is_first_batch=(batch_idx == 0))

            # Add processed contracts to final dataset
            for contract in featured_batch:
                processed_dataset.add(contract)

            # Memory cleanup
            del batch_dataset
            del featured_batch
            gc.collect()

            self.logger.info(f"[BATCH {batch_num}/{n_batches}] Completed")

        # Aggregate results
        self.results["dataset"] = processed_dataset

        # Merge microstructure summaries
        if all_microstructure_summaries:
            self.results["microstructure_summary"] = pd.concat(
                all_microstructure_summaries, ignore_index=True
            )

        # Merge trajectory results (take structure from first, metrics are per-contract anyway)
        if all_trajectory_results:
            self.results["event_trajectories"] = all_trajectory_results[0]

        # Free cleaned dataset
        del cleaned_dataset
        gc.collect()

        # STAGE 6: Visualization (optional, on aggregated data)
        if save_figures:
            self.logger.info("[VISUALIZATION] Starting...")
            figure_paths = self.visualization_stage.run(processed_dataset)
            self.results["figures"] = figure_paths
            self.logger.info(f"[VISUALIZATION] Complete - {len(figure_paths)} figures")

        # STAGE 7: Final export (category summaries, transitions)
        if export_tables:
            self.logger.info("[EXPORT] Finalizing exports...")
            self._finalize_exports(tables_dir)

        elapsed = datetime.now() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Batched pipeline completed in {elapsed.total_seconds():.2f}s")
        self.logger.info("=" * 60)

        self._print_summary()

        return self.results

    def _clear_batch_outputs(self, tables_dir: Path) -> None:
        """Clear existing batch output files for fresh run."""
        batch_files = [
            "contract_summary_extended.csv",
            "microstructure_summary.csv",
        ]
        for filename in batch_files:
            filepath = tables_dir / filename
            if filepath.exists():
                filepath.unlink()
                self.logger.debug(f"Cleared existing {filename}")

    def _export_batch_metrics(
        self,
        batch_dataset: MarketDataset,
        output_dir: Path,
        is_first_batch: bool,
    ) -> None:
        """Export batch metrics in append mode."""
        from src.utils.export import MetricsExporter

        exporter = MetricsExporter(self.config)

        # Contract summary - compute and append
        rows = []
        for contract in batch_dataset:
            metrics = exporter.compute_contract_metrics(contract)
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
            for regime, prop in metrics.regime_proportions.items():
                row[f"regime_pct_{regime}"] = prop * 100
            for lc_metric, value in metrics.lifecycle_metrics.items():
                row[f"lifecycle_{lc_metric}"] = value
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            output_path = output_dir / "contract_summary_extended.csv"

            # Write header only for first batch, append for subsequent
            if is_first_batch or not output_path.exists():
                df.to_csv(output_path, index=False, mode="w")
            else:
                df.to_csv(output_path, index=False, mode="a", header=False)

    def _finalize_exports(self, tables_dir: Path) -> None:
        """Finalize exports with category summaries and transitions."""
        from src.utils.export import MetricsExporter

        exporter = MetricsExporter(self.config)
        dataset = self.results.get("dataset")

        if dataset:
            # Category summary
            exporter.export_category_summary(
                dataset,
                tables_dir / "category_summary_extended.csv",
            )

            # Lifecycle binned
            exporter.export_lifecycle_binned(
                dataset,
                n_bins=50,
                output_path=tables_dir / "lifecycle_binned_metrics.csv",
            )

            # Regime transitions
            exporter.export_regime_transitions(
                dataset,
                tables_dir / "regime_transitions.csv",
            )

        # Microstructure summary
        if "microstructure_summary" in self.results:
            micro_path = tables_dir / "microstructure_summary.csv"
            self.results["microstructure_summary"].to_csv(micro_path, index=False)

        # Cleaning stats
        if "cleaning_stats" in self.results:
            stats_records = []
            for contract_id, stats in self.results["cleaning_stats"].items():
                stats_records.append({
                    "contract_id": contract_id,
                    "original_rows": stats.original_rows,
                    "final_rows": stats.final_rows,
                    "rows_removed": stats.rows_removed,
                    "removal_pct": stats.removal_pct,
                    "nulls_imputed": stats.nulls_imputed,
                    "outliers_winsorized": stats.outliers_winsorized,
                })
            if stats_records:
                stats_path = tables_dir / "cleaning_stats.csv"
                pd.DataFrame(stats_records).to_csv(stats_path, index=False)

        self.results["exports"] = {
            "contract_summary": tables_dir / "contract_summary_extended.csv",
            "category_summary": tables_dir / "category_summary_extended.csv",
            "lifecycle_binned": tables_dir / "lifecycle_binned_metrics.csv",
            "regime_transitions": tables_dir / "regime_transitions.csv",
            "microstructure_summary": tables_dir / "microstructure_summary.csv",
            "cleaning_stats": tables_dir / "cleaning_stats.csv",
        }

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        print("\n" + "=" * 60)
        print("BATCHED PIPELINE SUMMARY")
        print("=" * 60 + "\n")

        if "dataset" in self.results:
            dataset = self.results["dataset"]
            print(f"Contracts Processed: {len(dataset)}")
            print(f"Batch Size: {self.batch_size}")
            print(f"Categories: {', '.join(dataset.categories)}")

            for category in dataset.categories:
                cat_count = len(dataset.filter_by_category(category))
                print(f"  - {category}: {cat_count} contracts")

        if "microstructure_summary" in self.results:
            micro = self.results["microstructure_summary"]
            print(f"\nMicrostructure Metrics:")
            print(f"  - Avg Spread: {micro['avg_spread'].mean():.4f}")
            print(f"  - Avg Volume: {micro['avg_volume'].mean():.2f}")
            if "surge_count" in micro.columns:
                print(f"  - Total Surge Events: {micro['surge_count'].sum():.0f}")

        if "event_trajectories" in self.results:
            traj = self.results["event_trajectories"]
            print(f"\nEvent Trajectories:")
            print(f"  - Metrics Analyzed: {len(traj.get('trajectories', {}))}")
            print(f"  - Signature Curves: {len(traj.get('signatures', {}))}")

        if "figures" in self.results:
            print(f"\nFigures Generated: {len(self.results['figures'])}")

        if "exports" in self.results:
            print(f"\nTables Exported: {len(self.results['exports'])}")
            for name, path in self.results["exports"].items():
                print(f"  - {name}: {path}")

        print("\n" + "=" * 60 + "\n")


class ResearchPipeline:
    """
    Main research pipeline orchestrator.

    Coordinates all pipeline stages in sequence:
    ingestion → cleaning → features → microstructure → visualization → export
    """

    def __init__(
        self,
        config: ELTRConfig | None = None,
        categories: list[Category] | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize research pipeline.

        Parameters
        ----------
        config : ELTRConfig | None
            Platform configuration.
        categories : list[Category] | None
            Categories to analyze.
        output_dir : Path | None
            Output directory for results.
        """
        self.config = config or get_config()
        self.categories = categories
        self.output_dir = output_dir or Path(self.config.output.base_path)
        self.logger = get_logger("pipeline")

        self.stages = {
            "ingestion": IngestionStage(self.config, categories),
            "cleaning": CleaningStage(self.config),
            "features": FeatureStage(self.config),
            "microstructure": MicrostructureStage(self.config),
            "event_trajectories": EventTrajectoryStage(self.config),
            "visualization": VisualizationStage(self.config, self.output_dir / "figures"),
            "export": ExportStage(self.config, self.output_dir / "tables"),
        }

        self.results: dict[str, Any] = {}

    def run(
        self,
        stages: list[str] | None = None,
        save_figures: bool = True,
        export_tables: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the research pipeline.

        Parameters
        ----------
        stages : list[str] | None
            Stages to run. Runs all if None.
        save_figures : bool
            Generate and save figures.
        export_tables : bool
            Export summary tables.

        Returns
        -------
        dict[str, Any]
            Pipeline results.
        """
        self.logger.info("=" * 60)
        self.logger.info("KALSHI PREDICTION MARKET RESEARCH PIPELINE")
        self.logger.info("=" * 60)

        start_time = datetime.now()

        if stages is None:
            stages = ["ingestion", "cleaning", "features", "microstructure", "event_trajectories"]
            if save_figures:
                stages.append("visualization")
            if export_tables:
                stages.append("export")

        data = None

        for stage_name in stages:
            if stage_name not in self.stages:
                self.logger.warning(f"Unknown stage: {stage_name}")
                continue

            stage = self.stages[stage_name]
            self.logger.info(f"[{stage_name.upper()}] Starting...")

            if stage_name == "ingestion":
                data = stage.run()
                self.results["dataset"] = data

            elif stage_name == "cleaning":
                data, cleaning_stats = stage.run(data)
                self.results["dataset"] = data
                self.results["cleaning_stats"] = cleaning_stats

            elif stage_name == "features":
                data = stage.run(data)
                self.results["dataset"] = data

            elif stage_name == "microstructure":
                data, micro_summary = stage.run(data)
                self.results["microstructure_summary"] = micro_summary

            elif stage_name == "event_trajectories":
                trajectory_results = stage.run(data)
                self.results["event_trajectories"] = trajectory_results

            elif stage_name == "visualization":
                figure_paths = stage.run(data)
                self.results["figures"] = figure_paths

            elif stage_name == "export":
                export_paths = stage.run(self.results)
                self.results["exports"] = export_paths

            self.logger.info(f"[{stage_name.upper()}] Complete")

        elapsed = datetime.now() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {elapsed.total_seconds():.2f}s")
        self.logger.info("=" * 60)

        self._print_summary()

        return self.results

    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60 + "\n")

        if "dataset" in self.results:
            dataset = self.results["dataset"]
            print(f"Contracts Processed: {len(dataset)}")
            print(f"Categories: {', '.join(dataset.categories)}")

            for category in dataset.categories:
                cat_count = len(dataset.filter_by_category(category))
                print(f"  - {category}: {cat_count} contracts")

        if "microstructure_summary" in self.results:
            micro = self.results["microstructure_summary"]
            print(f"\nMicrostructure Metrics:")
            print(f"  - Avg Spread: {micro['avg_spread'].mean():.4f}")
            print(f"  - Avg Volume: {micro['avg_volume'].mean():.2f}")
            print(f"  - Total Surge Events: {micro['surge_count'].sum():.0f}")

        if "event_trajectories" in self.results:
            traj = self.results["event_trajectories"]
            print(f"\nEvent Trajectories:")
            print(f"  - Metrics Analyzed: {len(traj.get('trajectories', {}))}")
            print(f"  - Signature Curves: {len(traj.get('signatures', {}))}")

        if "figures" in self.results:
            print(f"\nFigures Generated: {len(self.results['figures'])}")

        if "exports" in self.results:
            print(f"\nTables Exported: {len(self.results['exports'])}")
            for name, path in self.results["exports"].items():
                print(f"  - {name}: {path}")

        print("\n" + "=" * 60 + "\n")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ELTR: Episodic Liquidity and Trading Regimes in Prediction Markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ELTR.py --full-analysis
    python ELTR.py --categories Sports,Politics --save-figures
    python ELTR.py --event-type economics --output-dir results
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to analyze",
    )

    parser.add_argument(
        "--event-type",
        type=str,
        default=None,
        help="Single event type to analyze (alias for --categories)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--save-figures",
        action="store_true",
        default=True,
        help="Generate and save visualization figures",
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        default=False,
        help="Skip figure generation",
    )

    parser.add_argument(
        "--export-tables",
        action="store_true",
        default=True,
        help="Export summary tables to CSV",
    )

    parser.add_argument(
        "--full-analysis",
        action="store_true",
        default=False,
        help="Run complete analysis pipeline with all options",
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help="Minimum observations required per contract",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Maximum number of contracts to process (for testing/memory limits)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of contracts to process per batch (default: 30)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    if args.verbose:
        config.logging.level = "DEBUG"

    if args.min_observations:
        config.cleaning.min_observations = args.min_observations

    setup_logging(config)
    logger = get_logger("main")

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    elif args.event_type:
        categories = [args.event_type.capitalize()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_figures = args.save_figures and not args.no_figures

    # Use BatchedResearchPipeline for memory-safe processing
    logger.info("Initializing batched research pipeline")
    logger.info(f"Batch size: {args.batch_size}")
    if args.max_contracts:
        logger.info(f"Max contracts: {args.max_contracts}")

    pipeline = BatchedResearchPipeline(
        config=config,
        categories=categories,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_contracts=args.max_contracts,
    )

    try:
        results = pipeline.run(
            save_figures=save_figures,
            export_tables=args.export_tables,
        )

        if "dataset" in results and len(results["dataset"]) > 0:
            logger.info("Pipeline completed successfully")
            return 0
        else:
            logger.warning("Pipeline completed but no data was processed")
            return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("KALSHI PREDICTION MARKET RESEARCH PLATFORM")
    print("=" * 60 + "\n")
    sys.exit(main())
