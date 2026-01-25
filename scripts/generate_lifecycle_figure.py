"""
Generate Figure 1: Lifecycle evolution of bid-ask spreads, trading volume, and volatility.

This script creates the 3-panel figure showing:
- Panel A: Median Bid-Ask Spread vs lifecycle position
- Panel B: Volume (p75) vs lifecycle position (log scale)
- Panel C: Volatility (p75) vs lifecycle position (log scale)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_lifecycle_figure(
    data_path: Path | str = "output/tables/lifecycle_binned_metrics.csv",
    output_path: Path | str = "F01_lifecycle_spread_volume_volatility.png",
    dpi: int = 300,
) -> None:
    """
    Generate the 3-panel lifecycle microstructure figure.

    Parameters
    ----------
    data_path : Path | str
        Path to lifecycle_binned_metrics.csv
    output_path : Path | str
        Output path for the figure
    dpi : int
        Figure resolution
    """
    # Load data
    df = pd.read_csv(data_path)

    # Aggregate across categories by lifecycle bin using mean
    # This captures the cross-category patterns better
    agg_data = df.groupby("lifecycle_bin").agg({
        "lifecycle_ratio": "first",
        "spread_median": "mean",  # Use mean of category medians
        "volume_p75": "mean",     # Use mean of category p75s
        "volatility_short_p75": "mean",  # Use mean of category p75s
    }).reset_index()

    # Sort by lifecycle position
    agg_data = agg_data.sort_values("lifecycle_ratio")

    # Handle zeros and NaN for log scale - replace with small positive value
    # for volume and volatility to show the pattern
    volume_floor = 1e-5
    volatility_floor = 1e-6

    # Replace zeros with floor values for log scale
    volume_data = agg_data["volume_p75"].copy()
    volume_data = volume_data.replace(0, volume_floor).fillna(volume_floor)
    # Ensure minimum value for log scale
    volume_data = volume_data.clip(lower=volume_floor)

    volatility_data = agg_data["volatility_short_p75"].copy()
    volatility_data = volatility_data.replace(0, volatility_floor).fillna(volatility_floor)
    volatility_data = volatility_data.clip(lower=volatility_floor)

    # Create figure with 3 vertically stacked panels
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)

    # Colors matching the reference figure
    spread_color = "#2c3e50"  # Dark blue-gray
    volume_color = "#27ae60"  # Green
    volatility_color = "#8b4513"  # Brown/maroon

    # Panel A: Bid-Ask Spread (median)
    ax1 = axes[0]
    ax1.plot(
        agg_data["lifecycle_ratio"],
        agg_data["spread_median"],
        color=spread_color,
        linewidth=1.5,
    )
    ax1.set_ylabel("Median Spread", fontsize=10)
    ax1.text(0.02, 0.92, "(A) Bid-Ask Spread", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax1.set_ylim(0, max(agg_data["spread_median"]) * 1.1)
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Volume (p75, log scale)
    ax2 = axes[1]
    ax2.plot(
        agg_data["lifecycle_ratio"],
        volume_data,
        color=volume_color,
        linewidth=1.5,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Volume (p75)", fontsize=10)
    ax2.text(0.02, 0.92, "(B) Volume", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: Volatility (p75, log scale)
    ax3 = axes[2]
    ax3.plot(
        agg_data["lifecycle_ratio"],
        volatility_data,
        color=volatility_color,
        linewidth=1.5,
    )
    ax3.set_yscale("log")
    ax3.set_ylabel("Volatility (p75)", fontsize=10)
    ax3.set_xlabel(r"Lifecycle Position ($\ell$)", fontsize=10)
    ax3.text(0.02, 0.92, "(C) Volatility", transform=ax3.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax3.grid(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(0, 1)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate lifecycle microstructure figure")
    parser.add_argument(
        "--data-path",
        type=str,
        default="output/tables/lifecycle_binned_metrics.csv",
        help="Path to lifecycle binned metrics CSV",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="F01_lifecycle_spread_volume_volatility.png",
        help="Output path for the figure",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )

    args = parser.parse_args()

    generate_lifecycle_figure(
        data_path=args.data_path,
        output_path=args.output_path,
        dpi=args.dpi,
    )
