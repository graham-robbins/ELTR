# IRP: Microstructure Analysis of Kalshi Prediction Markets

This repository contains the analysis codebase for studying market microstructure
dynamics in Kalshi prediction markets.

## Project Structure

```
src/
    microstructure/         # Core microstructure analysis modules
        spread.py           # Bid-ask spread computation and analysis
        liquidity.py        # Depth and resilience metrics
        bursts.py           # Surge and burst detection
        regimes.py          # State classification and Markov transitions
        lifecycle.py        # Contract lifecycle normalization
        analysis.py         # Orchestration layer (main entry point)
        event_alignment.py  # Event-aligned trajectory analysis
    features/               # Feature engineering pipelines
    utils/                  # Configuration, logging, and utilities

data/                       # Raw and processed datasets (excluded from VCS)
output/                     # Generated analysis artifacts (excluded from VCS)
final_tables/               # Publication-ready tables (excluded from VCS)
logs/                       # Runtime logs (excluded from VCS)
```

## Analysis Overview

The analysis pipeline processes OHLCV market data to compute:

- Spread dynamics and collapse trajectories over contract lifecycles
- Liquidity depth and resilience metrics via shock-recovery analysis
- Volume and volatility surge detection using z-score thresholds
- Microstructure regime classification (frozen, thin, normal, active, burst, resolution)
- Markov transition matrices for regime dynamics
- Event-aligned trajectory aggregation with percentile bands

## Data and Artifacts

Datasets and generated outputs are excluded from version control via `.gitignore`.
See `data/` for raw inputs and `output/` for computed results.

## References

See module docstrings for formal definitions of all computed metrics.
