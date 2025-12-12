"""
IRP: Kalshi Prediction Market Research Platform

Production-ready analytical framework for prediction market analysis.
"""

__version__ = "1.0.0"

from src.utils.config import IRPConfig, load_config, get_config
from src.utils.types import MarketDataset, ContractTimeseries

from src.ingest import load_kalshi_data
from src.clean import clean_market_data
from src.features import engineer_features
from src.microstructure import analyze_microstructure
from src.plots import generate_all_plots

__all__ = [
    "__version__",
    "IRPConfig",
    "load_config",
    "get_config",
    "MarketDataset",
    "ContractTimeseries",
    "load_kalshi_data",
    "clean_market_data",
    "engineer_features",
    "analyze_microstructure",
    "generate_all_plots",
]
