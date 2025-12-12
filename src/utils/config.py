"""
Configuration management for IRP prediction market analysis platform.

Provides centralized configuration loading, validation, and access patterns
following production engineering standards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Data source configuration."""
    base_path: str = "data/kalshi/kalshi_data_pull_v2"
    csv_output: str = "csv_output"
    metadata_file: str = "metadata.csv"
    schema_file: str = "schema_mapping.json"

    @property
    def csv_path(self) -> Path:
        return Path(self.base_path) / self.csv_output

    @property
    def metadata_path(self) -> Path:
        return Path(self.base_path) / self.metadata_file

    @property
    def schema_path(self) -> Path:
        return Path(self.base_path) / self.schema_file


@dataclass
class OutputConfig:
    """Output directory configuration."""
    base_path: str = "output"
    figures: str = "figures"
    tables: str = "tables"
    reports: str = "reports"

    @property
    def figures_path(self) -> Path:
        return Path(self.base_path) / self.figures

    @property
    def tables_path(self) -> Path:
        return Path(self.base_path) / self.tables

    @property
    def reports_path(self) -> Path:
        return Path(self.base_path) / self.reports


@dataclass
class CleaningConfig:
    """Data cleaning parameters."""
    min_observations: int = 50
    winsorize_limits: tuple[float, float] = (0.01, 0.01)
    max_spread_pct: float = 0.50
    min_price: int = 1
    max_price: int = 99
    max_gap_minutes: int = 60
    imputation_method: str = "forward_fill"
    drop_zero_volume_only: bool = False


@dataclass
class RollingWindowConfig:
    """Rolling window parameters."""
    short: int = 5
    medium: int = 15
    long: int = 60


@dataclass
class VolatilityConfig:
    """Volatility calculation parameters."""
    min_periods: int = 3
    annualization_factor: int = 525600


@dataclass
class LiquidityConfig:
    """Liquidity metric parameters."""
    depth_levels: list[int] = field(default_factory=lambda: [1, 5, 10])
    volume_ma_window: int = 20


@dataclass
class SpreadConfig:
    """Spread calculation parameters."""
    basis_points: bool = True


@dataclass
class LifecycleConfig:
    """Lifecycle feature parameters."""
    compute_tsl: bool = True
    compute_tts: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    rolling_windows: RollingWindowConfig = field(default_factory=RollingWindowConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)


@dataclass
class NormalizationConfig:
    """Amplitude normalization settings for cross-contract analysis."""
    spread: str = "zscore"
    volume: str = "percentile"
    depth: str = "minmax"
    volatility: str = "zscore"
    price: str = "minmax"
    default: str = "zscore"


@dataclass
class TimeBinningConfig:
    """Time binning configuration for cross-contract plots."""
    mode: str = "lifecycle"
    n_bins: int = 50
    lifecycle_bins: int = 100
    event_window_hours_pre: int = 24
    event_window_hours_post: int = 4


@dataclass
class EventAlignmentConfig:
    """Event alignment parameters."""
    pre_event_minutes: int = 60
    post_event_minutes: int = 30
    interpolation_grid_minutes: int = 1
    grid_hours_pre: int = 24
    grid_hours_post: int = 4


@dataclass
class SurgeDetectionConfig:
    """Surge detection parameters."""
    volume_threshold: float = 2.0
    volatility_threshold: float = 2.0
    lookback_window: int = 30


@dataclass
class RegimeDetectionConfig:
    """Regime detection parameters."""
    pregame_hours: int = 24
    halftime_buffer_minutes: int = 15


@dataclass
class RegimeClassificationConfig:
    """Microstructure regime classification thresholds."""
    frozen_volume_threshold: float = 0.1
    thin_spread_threshold: float = 0.15
    active_volume_zscore: float = 1.5
    burst_volatility_zscore: float = 2.0  # Legacy (unused)
    burst_volatility_k: float = 2.5  # Section 4: multiplier for volatility burst detection
    burst_volume_multiplier: float = 1.5  # Section 4: multiplier for volume condition
    resolution_lifecycle_threshold: float = 0.90  # Section 4: changed from 0.95 to 0.90
    resolution_spread_threshold: float = 0.05  # Section 4: spread threshold for resolution drift
    resolution_volume_quantile: float = 0.25  # Section 4: volume quantile threshold
    resolution_volatility_quantile: float = 0.25  # Section 4: volatility quantile threshold


@dataclass
class MicrostructureConfig:
    """Microstructure analysis configuration."""
    event_alignment: EventAlignmentConfig = field(default_factory=EventAlignmentConfig)
    surge_detection: SurgeDetectionConfig = field(default_factory=SurgeDetectionConfig)
    regime_detection: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    regime_classification: RegimeClassificationConfig = field(default_factory=RegimeClassificationConfig)


@dataclass
class FigsizeConfig:
    """Figure size configuration."""
    single: tuple[int, int] = (12, 6)
    quad: tuple[int, int] = (14, 10)
    heatmap: tuple[int, int] = (16, 12)


@dataclass
class PlotConfig:
    """Plotting configuration."""
    style: str = "seaborn-v0_8-whitegrid"
    figsize: FigsizeConfig = field(default_factory=FigsizeConfig)
    dpi: int = 150
    colormap: str = "RdYlGn"
    category_colors: dict[str, str] = field(default_factory=lambda: {
        "Sports": "#1f77b4",
        "Politics": "#ff7f0e",
        "Economics": "#2ca02c",
        "Weather": "#d62728",
        "Culture": "#9467bd",
    })
    save_format: str = "png"
    transparent: bool = False


@dataclass
class CorrelationConfig:
    """Correlation analysis parameters."""
    min_overlap: int = 20
    method: str = "pearson"


@dataclass
class StatisticsConfig:
    """Statistical analysis configuration."""
    confidence_level: float = 0.95
    bootstrap_resamples: int = 1000
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)


@dataclass
class TimezoneConfig:
    """Timezone configuration."""
    input: str = "UTC"
    output: str = "America/New_York"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/irp.log"


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    parallel_workers: int = 4
    chunk_size: int = 100
    cache_enabled: bool = True
    cache_dir: str = ".cache"


@dataclass
class IRPConfig:
    """
    Master configuration container for IRP platform.

    Aggregates all sub-configurations into a single access point.
    """
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    categories: list[str] = field(default_factory=lambda: [
        "Sports", "Politics", "Economics", "Weather", "Culture"
    ])
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    time_binning: TimeBinningConfig = field(default_factory=TimeBinningConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    timezone: TimezoneConfig = field(default_factory=TimezoneConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def _parse_nested_config(data: dict[str, Any], config_class: type) -> Any:
    """Parse nested configuration dictionaries into dataclass instances."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in config_class.__dataclass_fields__.values()}
    parsed = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
                parsed[key] = _parse_nested_config(value, field_type)
            elif isinstance(value, list) and key == "winsorize_limits":
                parsed[key] = tuple(value)
            elif isinstance(value, list) and key in ("single", "quad", "heatmap"):
                parsed[key] = tuple(value)
            else:
                parsed[key] = value

    return config_class(**parsed)


def load_config(config_path: str | Path | None = None) -> IRPConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to configuration file. If None, uses default config.

    Returns
    -------
    IRPConfig
        Loaded and validated configuration object.

    Raises
    ------
    FileNotFoundError
        If specified config file does not exist.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    config = IRPConfig()

    if "data" in raw_config:
        config.data = DataConfig(**raw_config["data"])

    if "output" in raw_config:
        config.output = OutputConfig(**raw_config["output"])

    if "categories" in raw_config:
        config.categories = raw_config["categories"]

    if "cleaning" in raw_config:
        cleaning_data = raw_config["cleaning"]
        if "winsorize_limits" in cleaning_data:
            cleaning_data["winsorize_limits"] = tuple(cleaning_data["winsorize_limits"])
        config.cleaning = CleaningConfig(**cleaning_data)

    if "features" in raw_config:
        feat = raw_config["features"]
        config.features = FeatureConfig(
            rolling_windows=RollingWindowConfig(**feat.get("rolling_windows", {})),
            volatility=VolatilityConfig(**feat.get("volatility", {})),
            liquidity=LiquidityConfig(**feat.get("liquidity", {})),
            spread=SpreadConfig(**feat.get("spread", {})),
            lifecycle=LifecycleConfig(**feat.get("lifecycle", {})),
        )

    if "normalization" in raw_config:
        config.normalization = NormalizationConfig(**raw_config["normalization"])

    if "time_binning" in raw_config:
        config.time_binning = TimeBinningConfig(**raw_config["time_binning"])

    if "microstructure" in raw_config:
        micro = raw_config["microstructure"]
        config.microstructure = MicrostructureConfig(
            event_alignment=EventAlignmentConfig(**micro.get("event_alignment", {})),
            surge_detection=SurgeDetectionConfig(**micro.get("surge_detection", {})),
            regime_detection=RegimeDetectionConfig(**micro.get("regime_detection", {})),
            regime_classification=RegimeClassificationConfig(**micro.get("regime_classification", {})),
        )

    if "plots" in raw_config:
        plot_data = raw_config["plots"]
        figsize_data = plot_data.pop("figsize", {})
        figsize = FigsizeConfig(
            single=tuple(figsize_data.get("single", [12, 6])),
            quad=tuple(figsize_data.get("quad", [14, 10])),
            heatmap=tuple(figsize_data.get("heatmap", [16, 12])),
        )
        config.plots = PlotConfig(figsize=figsize, **plot_data)

    if "statistics" in raw_config:
        stats = raw_config["statistics"]
        corr = stats.pop("correlation", {})
        config.statistics = StatisticsConfig(
            correlation=CorrelationConfig(**corr),
            **stats,
        )

    if "timezone" in raw_config:
        config.timezone = TimezoneConfig(**raw_config["timezone"])

    if "logging" in raw_config:
        config.logging = LoggingConfig(**raw_config["logging"])

    if "pipeline" in raw_config:
        config.pipeline = PipelineConfig(**raw_config["pipeline"])

    return config


_global_config: IRPConfig | None = None


def get_config() -> IRPConfig:
    """
    Get global configuration instance.

    Loads default configuration on first access.

    Returns
    -------
    IRPConfig
        Global configuration object.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: IRPConfig) -> None:
    """
    Set global configuration instance.

    Parameters
    ----------
    config : IRPConfig
        Configuration to set as global.
    """
    global _global_config
    _global_config = config
