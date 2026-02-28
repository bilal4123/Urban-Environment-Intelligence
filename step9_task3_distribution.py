import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTREME_THRESHOLD = 200  # µg/m³

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading final dataset...")
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")

    # ── Select industrial zone only ───────────────────────────────────────────
    print("\nFiltering industrial zone...")
    industrial = df[df["zone"] == "Industrial"]["PM25"].dropna()
    industrial = industrial[industrial > 0]
    print(f"  Industrial PM2.5 records: {len(industrial)}")
    print(f"  Max value: {industrial.max():.1f} µg/m³")
    print(f"  Mean value: {industrial.mean():.1f} µg/m³")

    # ── 99th percentile ───────────────────────────────────────────────────────
    p99 = np.percentile(industrial, 99)
    extreme_count = (industrial > EXTREME_THRESHOLD).sum()
    extreme_pct = (industrial > EXTREME_THRESHOLD).mean() * 100
    print(f"\n  99th percentile: {p99:.1f} µg/m³")
    print(f"  Records above {EXTREME_THRESHOLD} µg/m³: {extreme_count} ({extreme_pct:.3f}%)")

    # ── Figure: 2 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    # ── Panel 1: KDE — optimized to reveal PEAKS ─────────────────────────────
    ax1 = axes[0]
    # Use only values up to 300 for better peak visibility
    peak_data = industrial[industrial <= 300]

    kde = stats.gaussian_kde(peak_data, bw_method=0.1)
    x_range = np.linspace(0, 300, 1000)
    kde_values = kde(x_range)

    ax1.fill_between(x_range, kde_values, alpha=0.4, color="#457B9D")
    ax1.plot(x_range, kde_values, color="#457B9D", linewidth=1.5)

    # Mark health threshold
    ax1.axvline(35, color="#E63946", linewidth=1.5,
                linestyle="--", label="Health Threshold (35 µg/m³)")
    # Mark 99th percentile
    ax1.axvline(p99, color="#F4A261", linewidth=1.5,
                linestyle="--", label=f"99th Percentile ({p99:.0f} µg/m³)")

    ax1.set_xlabel("PM2.5 Concentration (µg/m³)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("KDE Plot — Optimized to Reveal Peaks\n(Industrial Zone PM2.5)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: ECDF — optimized to reveal TAILS ────────────────────────────
    ax2 = axes[1]

    # Complementary CDF (1 - CDF) on log scale reveals tail behavior
    sorted_data = np.sort(industrial)
    ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    ax2.plot(sorted_data, ccdf, color="#E63946", linewidth=1.2, alpha=0.85)

    # Mark extreme threshold
    ax2.axvline(EXTREME_THRESHOLD, color="#2D6A4F", linewidth=1.5,
                linestyle="--", label=f"Extreme Hazard ({EXTREME_THRESHOLD} µg/m³)")
    ax2.axvline(p99, color="#F4A261", linewidth=1.5,
                linestyle="--", label=f"99th Percentile ({p99:.0f} µg/m³)")

    # Annotate probability at extreme threshold
    prob_extreme = (industrial > EXTREME_THRESHOLD).mean()
    ax2.axhline(prob_extreme, color="#2D6A4F", linewidth=0.8,
                linestyle=":", alpha=0.7)
    ax2.annotate(
        f"P(PM2.5 > {EXTREME_THRESHOLD}) = {prob_extreme:.4f}\n({extreme_pct:.3f}%)",
        xy=(EXTREME_THRESHOLD, prob_extreme),
        xytext=(EXTREME_THRESHOLD + 50, prob_extreme + 0.02),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("PM2.5 Concentration (µg/m³) — Log Scale", fontsize=11)
    ax2.set_ylabel("P(X > x) — Log Scale", fontsize=11)
    ax2.set_title("Complementary CDF — Optimized to Reveal Tails\n(Industrial Zone PM2.5)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, frameon=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "task3_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved plot to {out_path}")

    # ── Analysis summary ──────────────────────────────────────────────────────
    print("\n── Analysis Summary ─────────────────────────────────────────────")
    print(f"  99th percentile PM2.5: {p99:.1f} µg/m³")
    print(f"  Probability of extreme hazard (>200 µg/m³): {prob_extreme:.4f} ({extreme_pct:.3f}%)")
    print(f"\n  KDE reveals: bulk of readings cluster near the mode,")
    print(f"  showing typical daily pollution levels clearly.")
    print(f"\n  CCDF reveals: the long tail extends well beyond 200 µg/m³,")
    print(f"  making rare extreme events visible that histograms would hide.")
    print(f"\n  Most honest plot for rare events: CCDF on log-log scale.")
    print(f"  Reason: log scale stretches the tail, making rare high values")
    print(f"  visible instead of compressed against the x-axis.")

if __name__ == "__main__":
    main()