"""
Microstructure analysis module for ELTR platform.

Provides market microstructure analytics and event analysis.

Module Structure:
    - spread.py: Bid-ask spread computation and analysis
    - liquidity.py: Depth and resilience metrics
    - bursts.py: Surge and burst detection
    - regimes.py: State classification and Markov transitions
    - lifecycle.py: Contract lifecycle normalization
    - analysis.py: Orchestration layer (backward compatible)
    - event_alignment.py: Event-aligned trajectory analysis
"""

# Core orchestration and backward-compatible classes
from src.microstructure.analysis import (
    MicrostructureAnalyzer,
    SurgeDetector,
    SpreadAnalyzer,
    TrajectoryAnalyzer,
    DepthAnalyzer,
    PrePostAnalyzer,
    SurgeEvent,
    EventWindow,
    MicrostructureMetrics,
    analyze_microstructure,
    run_full_analysis,
)

# Spread module
from src.microstructure.spread import (
    compute_spread_features,
    compute_effective_spread,
    compute_spread_collapse_slope,
    compute_spread_curve,
    compute_spread_percentiles,
    compute_spread_regime_stats,
)

# Liquidity module
from src.microstructure.liquidity import (
    compute_volume_features,
    compute_depth_features,
    compute_depth_resilience,
    compute_depth_impact,
    compute_liquidity_resilience,
)

# Bursts module
from src.microstructure.bursts import (
    detect_volume_surges,
    detect_volatility_surges,
    compute_surge_ratio,
    classify_volatility_burst,
    compute_burst_intensity,
)

# Regimes module
from src.microstructure.regimes import (
    MicrostructureState,
    compute_event_regime,
    compute_microstructure_regime,
    compute_transition_matrix,
    compute_regime_entropy,
)

# Lifecycle module
from src.microstructure.lifecycle import (
    LIFECYCLE_BIN_EDGES,
    LIFECYCLE_PHASE_NAMES,
    compute_lifecycle_features,
    make_lifecycle_bins,
    assign_lifecycle_bins,
    compute_binned_trajectory,
)

# Event alignment (unchanged)
from src.microstructure.event_alignment import (
    AlignedTrajectory,
    AggregatedTrajectory,
    EventAligner,
    TrajectoryAggregator,
    EventTrajectoryAnalyzer,
    compute_event_trajectories,
)

__all__ = [
    # Orchestration classes (backward compatible)
    "MicrostructureAnalyzer",
    "SurgeDetector",
    "SpreadAnalyzer",
    "TrajectoryAnalyzer",
    "DepthAnalyzer",
    "PrePostAnalyzer",
    "SurgeEvent",
    "EventWindow",
    "MicrostructureMetrics",
    "analyze_microstructure",
    "run_full_analysis",
    # Spread functions
    "compute_spread_features",
    "compute_effective_spread",
    "compute_spread_collapse_slope",
    "compute_spread_curve",
    "compute_spread_percentiles",
    "compute_spread_regime_stats",
    # Liquidity functions
    "compute_volume_features",
    "compute_depth_features",
    "compute_depth_resilience",
    "compute_depth_impact",
    "compute_liquidity_resilience",
    # Burst functions
    "detect_volume_surges",
    "detect_volatility_surges",
    "compute_surge_ratio",
    "classify_volatility_burst",
    "compute_burst_intensity",
    # Regime functions
    "MicrostructureState",
    "compute_event_regime",
    "compute_microstructure_regime",
    "compute_transition_matrix",
    "compute_regime_entropy",
    # Lifecycle functions
    "LIFECYCLE_BIN_EDGES",
    "LIFECYCLE_PHASE_NAMES",
    "compute_lifecycle_features",
    "make_lifecycle_bins",
    "assign_lifecycle_bins",
    "compute_binned_trajectory",
    # Event alignment
    "AlignedTrajectory",
    "AggregatedTrajectory",
    "EventAligner",
    "TrajectoryAggregator",
    "EventTrajectoryAnalyzer",
    "compute_event_trajectories",
]
