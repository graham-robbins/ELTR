"""
Type definitions and data structures for ELTR platform.

Provides typed containers, enums, and protocol definitions for
consistent data handling across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd


# Type aliases for clarity
ContractID: TypeAlias = str
Category: TypeAlias = str
Price: TypeAlias = float
Volume: TypeAlias = int
Timestamp: TypeAlias = pd.Timestamp


class MarketStatus(Enum):
    """Market lifecycle status."""
    OPEN = auto()
    CLOSED = auto()
    SETTLED = auto()
    FINALIZED = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, status: str) -> MarketStatus:
        """Parse status string to enum."""
        mapping = {
            "open": cls.OPEN,
            "closed": cls.CLOSED,
            "settled": cls.SETTLED,
            "finalized": cls.FINALIZED,
        }
        return mapping.get(status.lower(), cls.UNKNOWN)


class EventRegime(Enum):
    """Event-relative time regime classification."""
    PRE_EVENT = auto()
    PREGAME = auto()
    IN_GAME = auto()
    HALFTIME = auto()
    POST_EVENT = auto()
    EVENT_RELEASE = auto()
    UNKNOWN = auto()


class DataFrequency(Enum):
    """Data sampling frequency."""
    SECOND = auto()
    MINUTE = auto()
    HOUR = auto()
    DAY = auto()

    @property
    def pandas_freq(self) -> str:
        """Return pandas frequency string."""
        mapping = {
            self.SECOND: "s",
            self.MINUTE: "min",
            self.HOUR: "h",
            self.DAY: "D",
        }
        return mapping[self]


@dataclass
class OHLCV:
    """OHLCV bar representation."""
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int

    @property
    def is_valid(self) -> bool:
        """Check if bar has valid price data."""
        return self.close is not None

    @property
    def range(self) -> float | None:
        """Calculate high-low range."""
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None


@dataclass
class BidAsk:
    """Bid-ask quote representation."""
    bid: float | None
    ask: float | None

    @property
    def spread(self) -> float | None:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def midpoint(self) -> float | None:
        """Calculate midpoint price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread_pct(self) -> float | None:
        """Calculate spread as percentage of midpoint."""
        mid = self.midpoint
        spread = self.spread
        if mid is not None and spread is not None and mid > 0:
            return spread / mid
        return None


@dataclass
class ContractMetadata:
    """Contract-level metadata."""
    contract_id: ContractID
    category: Category
    event_ticker: str
    title: str
    status: MarketStatus
    start_date: datetime | None
    end_date: datetime | None
    resolution_rule: str
    total_volume: int
    open_interest: int
    last_price: float | None

    @classmethod
    def from_series(cls, row: pd.Series) -> ContractMetadata:
        """Construct from pandas Series."""
        return cls(
            contract_id=row.get("contract_id", ""),
            category=row.get("pm_category", row.get("category", "Unknown")),
            event_ticker=row.get("event_ticker", ""),
            title=row.get("title", ""),
            status=MarketStatus.from_string(str(row.get("status", ""))),
            start_date=pd.to_datetime(row.get("start_date"), errors="coerce"),
            end_date=pd.to_datetime(row.get("end_date"), errors="coerce"),
            resolution_rule=row.get("resolution_rule", ""),
            total_volume=int(row.get("volume", 0) or 0),
            open_interest=int(row.get("open_interest", 0) or 0),
            last_price=float(row["last_price"]) if pd.notna(row.get("last_price")) else None,
        )


@dataclass
class TimeseriesBar:
    """Single timeseries observation."""
    timestamp: Timestamp
    contract_id: ContractID
    price: OHLCV
    quote: BidAsk
    trade_count: int | None


@dataclass
class ContractTimeseries:
    """
    Container for contract timeseries data.

    Holds both raw DataFrame and metadata for a single contract.
    """
    contract_id: ContractID
    category: Category
    data: pd.DataFrame
    metadata: ContractMetadata | None = None
    frequency: DataFrequency = DataFrequency.MINUTE
    event_time: Timestamp | None = None

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self.data)

    @property
    def date_range(self) -> tuple[Timestamp, Timestamp] | None:
        """Start and end timestamps."""
        if self.data.empty:
            return None
        return self.data.index.min(), self.data.index.max()

    @property
    def has_trades(self) -> bool:
        """Check if contract has any trades."""
        if "volume" not in self.data.columns:
            return False
        return self.data["volume"].sum() > 0

    @property
    def duration_hours(self) -> float | None:
        """Contract duration in hours (Section 13)."""
        if self.data.empty:
            return None
        start, end = self.date_range
        return (end - start).total_seconds() / 3600


@dataclass
class MarketDataset:
    """
    Container for multiple contract timeseries.

    Primary data structure for pipeline operations.
    """
    contracts: dict[ContractID, ContractTimeseries] = field(default_factory=dict)
    metadata: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.contracts)

    def __iter__(self):
        return iter(self.contracts.values())

    def __getitem__(self, contract_id: ContractID) -> ContractTimeseries:
        return self.contracts[contract_id]

    def add(self, contract: ContractTimeseries) -> None:
        """Add contract to dataset."""
        self.contracts[contract.contract_id] = contract

    def remove(self, contract_id: ContractID) -> None:
        """Remove contract from dataset."""
        if contract_id in self.contracts:
            del self.contracts[contract_id]

    def filter_by_category(self, category: Category) -> MarketDataset:
        """Return subset filtered by category."""
        filtered = MarketDataset(metadata=self.metadata)
        for contract in self.contracts.values():
            if contract.category == category:
                filtered.add(contract)
        return filtered

    def filter_by_min_observations(self, min_obs: int) -> MarketDataset:
        """Return subset with minimum observations."""
        filtered = MarketDataset(metadata=self.metadata)
        for contract in self.contracts.values():
            if contract.n_observations >= min_obs:
                filtered.add(contract)
        return filtered

    @property
    def categories(self) -> list[Category]:
        """Unique categories in dataset."""
        return list(set(c.category for c in self.contracts.values()))

    @property
    def contract_ids(self) -> list[ContractID]:
        """List of all contract IDs."""
        return list(self.contracts.keys())

    def to_panel(self) -> pd.DataFrame:
        """
        Convert to panel DataFrame with MultiIndex (contract_id, timestamp).
        """
        frames = []
        for contract in self.contracts.values():
            df = contract.data.copy()
            df["contract_id"] = contract.contract_id
            df["category"] = contract.category
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=False)
        panel = panel.reset_index()
        panel = panel.set_index(["contract_id", "timestamp"])
        return panel.sort_index()

    def summary(self) -> pd.DataFrame:
        """Generate summary statistics for all contracts."""
        records = []
        for contract in self.contracts.values():
            record = {
                "contract_id": contract.contract_id,
                "category": contract.category,
                "n_observations": contract.n_observations,
                "frequency": contract.frequency.name,
            }
            date_range = contract.date_range
            if date_range:
                record["start_date"] = date_range[0]
                record["end_date"] = date_range[1]
            if "volume" in contract.data.columns:
                record["total_volume"] = contract.data["volume"].sum()
            if "price_c" in contract.data.columns:
                record["mean_price"] = contract.data["price_c"].mean()
            records.append(record)

        return pd.DataFrame(records)


@dataclass
class AnalyticsResult:
    """Container for analytics output."""
    name: str
    contract_id: ContractID | None
    category: Category | None
    data: pd.DataFrame | pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


class DataTransformer(Protocol):
    """Protocol for data transformation operations."""

    def transform(self, data: MarketDataset) -> MarketDataset:
        """Apply transformation to dataset."""
        ...


class FeatureExtractor(Protocol):
    """Protocol for feature extraction operations."""

    def extract(self, data: ContractTimeseries) -> pd.DataFrame:
        """Extract features from timeseries."""
        ...


class Analyzer(Protocol):
    """Protocol for analysis operations."""

    def analyze(self, data: MarketDataset) -> AnalyticsResult:
        """Run analysis on dataset."""
        ...
