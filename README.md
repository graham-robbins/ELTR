# ELTR: Episodic Liquidity and Trading Regimes in Prediction Markets

Implementation of microstructure analysis methods from "Episodic Liquidity and Trading Regimes in Prediction Markets" (Robbins & Taillard).

## Overview

This codebase analyzes high-frequency order book data from Kalshi prediction markets to characterize how trading activity and liquidity evolve over contract lifecycles. The analysis classifies market microstructure into discrete regimes and models regime dynamics as a Markov process.

## Project Structure

```
src/
    microstructure/
        spread.py           # Bid-ask spread computation (absolute and normalized)
        liquidity.py        # Depth resilience and shock recovery metrics
        bursts.py           # Volume surprise and volatility burst detection
        regimes.py          # Six-state classification and Markov transitions
        lifecycle.py        # Contract lifecycle normalization (t̃ ∈ [0,1])
        analysis.py         # Pipeline orchestration
        event_alignment.py  # Event-aligned trajectory aggregation
    features/               # Feature engineering pipelines
    utils/                  # Configuration and logging

data/                       # Raw datasets (excluded from VCS)
output/                     # Analysis artifacts (excluded from VCS)
```

## Microstructure States

The classification assigns each observation to one of six mutually exclusive states:

| State | Description |
|-------|-------------|
| **Frozen** | No trading; price discovery stalled |
| **Thin** | Scarce trading with wide bid-ask spreads |
| **Normal** | Baseline conditions with moderate activity |
| **Active Information** | Elevated volume consistent with information arrival |
| **Volatility Burst** | Sharp price movements with high volume |
| **Resolution Drift** | Late-stage quiet trading as uncertainty collapses |

States are assigned deterministically via priority ordering: Frozen > Volatility Burst > Resolution Drift > Active Information > Thin > Normal.

## Key Metrics

- **Lifecycle position** (`t̃`): Normalized time from listing to resolution
- **Normalized spread** (`s̃`): Cross-contract comparable spread measure
- **Volume surprise** (`σ̂`): Standardized deviation from rolling baseline
- **Regime entropy** (`H^norm`): Diversity of states over contract lifetime

## Data

Datasets are excluded from version control. Place raw Kalshi order book data in `data/` and run the analysis pipeline to generate outputs.
