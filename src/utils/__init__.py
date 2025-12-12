"""
Utility modules for IRP platform.

Provides configuration, types, logging, normalization,
time binning, and export infrastructure.
"""

from src.utils.config import (
    IRPConfig,
    load_config,
    get_config,
    set_config,
)
from src.utils.types import (
    ContractID,
    Category,
    Price,
    Volume,
    Timestamp,
    MarketStatus,
    EventRegime,
    DataFrequency,
    OHLCV,
    BidAsk,
    ContractMetadata,
    ContractTimeseries,
    MarketDataset,
    AnalyticsResult,
)
from src.utils.logging import setup_logging, get_logger
from src.utils.normalization import (
    normalize_series,
    normalize_dataframe,
    normalize_cross_contract,
    Normalizer,
)
from src.utils.time_binning import (
    compute_lifecycle_features,
    make_lifecycle_bins,
    make_tts_bins,
    make_event_aligned_bins,
    make_time_bins,
    aggregate_by_bins,
    compute_binned_trajectory,
)
from src.utils.export import (
    MetricsExporter,
    ContractSummaryMetrics,
    CategorySummaryMetrics,
    export_metrics,
)

__all__ = [
    # Config
    "IRPConfig",
    "load_config",
    "get_config",
    "set_config",
    # Types
    "ContractID",
    "Category",
    "Price",
    "Volume",
    "Timestamp",
    "MarketStatus",
    "EventRegime",
    "DataFrequency",
    "OHLCV",
    "BidAsk",
    "ContractMetadata",
    "ContractTimeseries",
    "MarketDataset",
    "AnalyticsResult",
    # Logging
    "setup_logging",
    "get_logger",
    # Normalization
    "normalize_series",
    "normalize_dataframe",
    "normalize_cross_contract",
    "Normalizer",
    # Time binning
    "compute_lifecycle_features",
    "make_lifecycle_bins",
    "make_tts_bins",
    "make_event_aligned_bins",
    "make_time_bins",
    "aggregate_by_bins",
    "compute_binned_trajectory",
    # Export
    "MetricsExporter",
    "ContractSummaryMetrics",
    "CategorySummaryMetrics",
    "export_metrics",
]
