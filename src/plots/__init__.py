"""
Visualization module for ELTR platform.

Provides plotting capabilities for prediction market analysis.

Research plot types include:
- Spread Tightening Curves
- Liquidity Trajectories
- Volatility Burst Maps
- Regime Occupancy Diagrams
- Category Fingerprints
- Event-Aligned Trajectories
"""

from src.plots.plotting import (
    PlotManager,
    BasePlotter,
    TimeSeriesPlotter,
    HeatmapPlotter,
    CategoryPlotter,
    TrajectoryPlotter,
    SpreadEvolutionPlotter,
    DepthPlotter,
    SpreadTighteningPlotter,
    LiquidityTrajectoryPlotter,
    VolatilityBurstPlotter,
    RegimeOccupancyPlotter,
    CategoryFingerprintPlotter,
    EventTrajectoryPlotter,
    generate_all_plots,
)

__all__ = [
    "PlotManager",
    "BasePlotter",
    "TimeSeriesPlotter",
    "HeatmapPlotter",
    "CategoryPlotter",
    "TrajectoryPlotter",
    "SpreadEvolutionPlotter",
    "DepthPlotter",
    "SpreadTighteningPlotter",
    "LiquidityTrajectoryPlotter",
    "VolatilityBurstPlotter",
    "RegimeOccupancyPlotter",
    "CategoryFingerprintPlotter",
    "EventTrajectoryPlotter",
    "generate_all_plots",
]
