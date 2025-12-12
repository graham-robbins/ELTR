"""
Data ingestion module for IRP platform.

Provides loaders for Kalshi prediction market data.
"""

from src.ingest.kalshi_loader import (
    KalshiDataLoader,
    ContractLoader,
    MetadataLoader,
    SchemaValidator,
    TimestampParser,
    load_kalshi_data,
)

__all__ = [
    "KalshiDataLoader",
    "ContractLoader",
    "MetadataLoader",
    "SchemaValidator",
    "TimestampParser",
    "load_kalshi_data",
]
