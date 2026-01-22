"""
Robustness Analysis for Section 5.1

Runs regime classification under multiple threshold specifications
to verify that findings are not artifacts of particular parameter choices.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Import from existing codebase
from src.microstructure.regimes import MicrostructureState


def load_all_contracts(data_dir: Path, min_observations: int = 100) -> List[Tuple[str, str, pd.DataFrame]]:
    """Load all contracts from CSV files."""
    contracts = []
    csv_dir = data_dir / "kalshi" / "kalshi_data_pull_v2" / "csv_output"

    for category_dir in csv_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for csv_file in category_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, parse_dates=["datetime"])
                df = df.set_index("datetime").sort_index()

                if len(df) < min_observations:
                    continue

                # Basic feature engineering
                df = prepare_features(df)
                if df is not None and len(df) >= min_observations:
                    contracts.append((csv_file.stem, category, df))
            except Exception as e:
                continue

    return contracts


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features needed for regime classification."""
    df = df.copy()

    # Spread
    if "yes_bid_c" in df.columns and "yes_ask_c" in df.columns:
        df["spread"] = df["yes_ask_c"] - df["yes_bid_c"]
        df["spread"] = df["spread"].clip(lower=0)

        # Midprice
        df["midprice"] = (df["yes_bid_c"] + df["yes_ask_c"]) / 2
        df["midprice"] = df["midprice"].replace(0, np.nan)

        # Spread percentage
        df["spread_pct"] = df["spread"] / df["midprice"]
        df["spread_pct"] = df["spread_pct"].clip(0, 1)

    # Volume - fill NaN with 0
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # Price and returns
    if "price_c" in df.columns:
        df["price_c"] = df["price_c"].ffill()
        df["pct_return"] = df["price_c"].pct_change()
        df["pct_return"] = df["pct_return"].fillna(0)

    # Lifecycle ratio
    if len(df) > 0:
        start_time = df.index.min()
        end_time = df.index.max()
        total_duration = (end_time - start_time).total_seconds()
        if total_duration > 0:
            elapsed = (df.index - start_time).total_seconds()
            df["lifecycle_ratio"] = elapsed / total_duration
        else:
            df["lifecycle_ratio"] = 0.5

    # Drop rows with missing essential data
    df = df.dropna(subset=["spread_pct", "volume"], how="all")

    return df


def classify_regimes(
    df: pd.DataFrame,
    frozen_volume_threshold: float = 0.1,
    thin_spread_threshold: float = 0.15,
    active_volume_zscore: float = 1.5,
    burst_volatility_k: float = 2.5,
    burst_volume_multiplier: float = 1.5,
    resolution_lifecycle_threshold: float = 0.90,
    rolling_window: int = 20,
) -> pd.Series:
    """
    Classify microstructure state for each observation.
    Simplified version of compute_microstructure_regime for this analysis.
    """
    states = pd.Series(MicrostructureState.NORMAL.value, index=df.index)

    # Compute rolling metrics
    raw_rolling_volatility = None
    raw_rolling_volume = None

    if "pct_return" in df.columns:
        raw_rolling_volatility = df["pct_return"].abs().rolling(
            window=rolling_window, min_periods=3
        ).mean()

    if "volume" in df.columns:
        raw_rolling_volume = df["volume"].rolling(
            window=rolling_window, min_periods=1
        ).mean()

    # Volatility burst detection
    if raw_rolling_volatility is not None and "pct_return" in df.columns:
        mid_return = df["pct_return"].abs()
        volatility_condition = mid_return > (burst_volatility_k * raw_rolling_volatility)

        if raw_rolling_volume is not None:
            volume_condition = df["volume"] > (raw_rolling_volume * burst_volume_multiplier)
            burst_mask = volatility_condition & volume_condition
        else:
            burst_mask = volatility_condition

        states[burst_mask] = MicrostructureState.VOLATILITY_BURST.value

    # Active information arrival
    if raw_rolling_volume is not None and "volume" in df.columns:
        vol_std = df["volume"].rolling(window=rolling_window, min_periods=1).std()
        vol_zscore = np.where(
            vol_std > 0,
            (df["volume"] - raw_rolling_volume) / vol_std,
            0,
        )
        active_volume_mask = vol_zscore > active_volume_zscore
        active_info_mask = active_volume_mask & (states == MicrostructureState.NORMAL.value)
        states[active_info_mask] = MicrostructureState.ACTIVE_INFORMATION.value

    # Resolution drift
    if "lifecycle_ratio" in df.columns:
        lifecycle_mask = df["lifecycle_ratio"] > resolution_lifecycle_threshold

        spread_ok = pd.Series(True, index=df.index)
        volume_ok = pd.Series(True, index=df.index)
        volatility_ok = pd.Series(True, index=df.index)

        if "spread_pct" in df.columns:
            spread_threshold = df["spread_pct"].quantile(0.05)
            spread_ok = df["spread_pct"] < max(spread_threshold, 0.05)

        if "volume" in df.columns:
            volume_threshold = df["volume"].quantile(0.25)
            volume_ok = df["volume"] < volume_threshold

        if raw_rolling_volatility is not None:
            vol_threshold = raw_rolling_volatility.quantile(0.25)
            volatility_ok = raw_rolling_volatility < vol_threshold

        resolution_mask = lifecycle_mask & spread_ok & volume_ok & volatility_ok
        resolution_mask = resolution_mask & (states != MicrostructureState.VOLATILITY_BURST.value)
        states[resolution_mask] = MicrostructureState.RESOLUTION_DRIFT.value

    # Thin market
    if "spread_pct" in df.columns:
        thin_mask = df["spread_pct"] > thin_spread_threshold
        thin_mask = thin_mask & (states == MicrostructureState.NORMAL.value)
        states[thin_mask] = MicrostructureState.THIN.value

    # Frozen market (highest priority - applied last)
    frozen_mask = pd.Series(False, index=df.index)

    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(window=rolling_window, min_periods=1).mean()
        frozen_mask = frozen_mask | (df["volume"] < (vol_ma * frozen_volume_threshold))
        frozen_mask = frozen_mask | (df["volume"] == 0)

    if "pct_return" in df.columns:
        frozen_mask = frozen_mask | (df["pct_return"] == 0)

    states[frozen_mask] = MicrostructureState.FROZEN.value

    return states


def compute_regime_proportions(states: pd.Series) -> Dict[str, float]:
    """Compute proportion of time in each regime."""
    counts = states.value_counts()
    total = len(states)

    state_names = {
        MicrostructureState.FROZEN.value: "Frozen",
        MicrostructureState.THIN.value: "Thin",
        MicrostructureState.NORMAL.value: "Normal",
        MicrostructureState.ACTIVE_INFORMATION.value: "Active Info",
        MicrostructureState.VOLATILITY_BURST.value: "Vol Burst",
        MicrostructureState.RESOLUTION_DRIFT.value: "Res Drift",
    }

    props = {}
    for state_val, state_name in state_names.items():
        props[state_name] = counts.get(state_val, 0) / total * 100

    return props


def run_robustness_analysis(contracts: List[Tuple[str, str, pd.DataFrame]]) -> pd.DataFrame:
    """Run regime classification under multiple specifications."""

    # Define specifications to test
    specifications = {
        "Baseline": {
            "frozen_volume_threshold": 0.10,
            "thin_spread_threshold": 0.15,
            "active_volume_zscore": 1.5,
            "burst_volatility_k": 2.5,
            "burst_volume_multiplier": 1.5,
            "rolling_window": 20,
        },
        # Frozen threshold variations
        "θ_F = 0.05 (-50%)": {"frozen_volume_threshold": 0.05},
        "θ_F = 0.15 (+50%)": {"frozen_volume_threshold": 0.15},
        # Thin threshold variations
        "θ_T = 0.075 (-50%)": {"thin_spread_threshold": 0.075},
        "θ_T = 0.225 (+50%)": {"thin_spread_threshold": 0.225},
        # Active info threshold variations
        "θ_A = 1.0 (-33%)": {"active_volume_zscore": 1.0},
        "θ_A = 2.0 (+33%)": {"active_volume_zscore": 2.0},
        # Burst threshold variations
        "κ = 1.5 (-40%)": {"burst_volatility_k": 1.5},
        "κ = 3.5 (+40%)": {"burst_volatility_k": 3.5},
        # Rolling window variations
        "W = 10 (-50%)": {"rolling_window": 10},
        "W = 30 (+50%)": {"rolling_window": 30},
    }

    baseline = specifications["Baseline"]
    results = []

    for spec_name, spec_changes in specifications.items():
        print(f"Running specification: {spec_name}")

        # Merge with baseline
        params = baseline.copy()
        if spec_name != "Baseline":
            params.update(spec_changes)

        # Accumulate all observations across contracts
        all_states = []

        for contract_id, category, df in contracts:
            try:
                states = classify_regimes(df, **params)
                all_states.append(states)
            except Exception as e:
                continue

        if all_states:
            # Combine all states
            combined_states = pd.concat(all_states)
            props = compute_regime_proportions(combined_states)

            row = {"Specification": spec_name}
            row.update(props)
            row["N_obs"] = len(combined_states)
            results.append(row)

    df_results = pd.DataFrame(results)

    # Reorder columns
    col_order = ["Specification", "Frozen", "Thin", "Normal", "Active Info", "Vol Burst", "Res Drift", "N_obs"]
    df_results = df_results[[c for c in col_order if c in df_results.columns]]

    return df_results


def run_bootstrap_analysis(
    contracts: List[Tuple[str, str, pd.DataFrame]],
    n_bootstrap: int = 100
) -> pd.DataFrame:
    """Bootstrap confidence intervals by resampling contracts."""

    baseline_params = {
        "frozen_volume_threshold": 0.10,
        "thin_spread_threshold": 0.15,
        "active_volume_zscore": 1.5,
        "burst_volatility_k": 2.5,
        "burst_volume_multiplier": 1.5,
        "rolling_window": 20,
    }

    bootstrap_results = defaultdict(list)
    n_contracts = len(contracts)

    print(f"Running {n_bootstrap} bootstrap replications...")

    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap {i + 1}/{n_bootstrap}")

        # Resample contracts with replacement
        indices = np.random.choice(n_contracts, size=n_contracts, replace=True)
        sampled_contracts = [contracts[idx] for idx in indices]

        # Classify all observations
        all_states = []
        for contract_id, category, df in sampled_contracts:
            try:
                states = classify_regimes(df, **baseline_params)
                all_states.append(states)
            except:
                continue

        if all_states:
            combined = pd.concat(all_states)
            props = compute_regime_proportions(combined)

            for regime, prop in props.items():
                bootstrap_results[regime].append(prop)

    # Compute confidence intervals
    ci_results = []
    for regime, values in bootstrap_results.items():
        values = np.array(values)
        ci_results.append({
            "Regime": regime,
            "Mean": np.mean(values),
            "Std": np.std(values),
            "CI_2.5%": np.percentile(values, 2.5),
            "CI_97.5%": np.percentile(values, 97.5),
        })

    return pd.DataFrame(ci_results)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading contracts...")
    contracts = load_all_contracts(data_dir, min_observations=100)
    print(f"Loaded {len(contracts)} contracts")

    # Count by category
    category_counts = defaultdict(int)
    for _, cat, _ in contracts:
        category_counts[cat] += 1
    print("By category:", dict(category_counts))

    # Run robustness analysis
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS")
    print("="*60)

    results = run_robustness_analysis(contracts)

    print("\n" + "="*60)
    print("TABLE: Regime Proportions Under Alternative Specifications")
    print("="*60)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Save to CSV
    results.to_csv(output_dir / "robustness_table.csv", index=False)
    print(f"\nSaved to {output_dir / 'robustness_table.csv'}")

    # Run bootstrap
    print("\n" + "="*60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*60)

    ci_results = run_bootstrap_analysis(contracts, n_bootstrap=100)
    print(ci_results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    ci_results.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    print(f"\nSaved to {output_dir / 'bootstrap_ci.csv'}")

    # Generate LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)

    latex = generate_latex_table(results)
    print(latex)

    with open(output_dir / "robustness_table.tex", "w") as f:
        f.write(latex)
    print(f"\nSaved to {output_dir / 'robustness_table.tex'}")


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table code."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Regime Proportions Under Alternative Threshold Specifications}
\label{tab:robustness}
\begin{tabular}{lcccccc}
\toprule
Specification & Frozen & Thin & Normal & Active Info & Vol Burst & Res Drift \\
\midrule
"""

    for _, row in df.iterrows():
        spec = row["Specification"]
        frozen = row.get("Frozen", 0)
        thin = row.get("Thin", 0)
        normal = row.get("Normal", 0)
        active = row.get("Active Info", 0)
        burst = row.get("Vol Burst", 0)
        drift = row.get("Res Drift", 0)

        latex += f"{spec} & {frozen:.1f}\\% & {thin:.1f}\\% & {normal:.1f}\\% & {active:.1f}\\% & {burst:.1f}\\% & {drift:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Each row reports the percentage of observations classified into each microstructure state under the specified parameterization. Baseline parameters: $\theta_F = 0.10$, $\theta_T = 0.15$, $\theta_A = 1.5$, $\kappa = 2.5$, $\lambda = 1.5$, $W = 20$. Alternative specifications vary one parameter at a time while holding others at baseline values.
\end{tablenotes}
\end{table}
"""

    return latex


if __name__ == "__main__":
    main()
