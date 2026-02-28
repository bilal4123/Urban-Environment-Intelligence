import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 35  # PM2.5 health threshold µg/m³

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading final dataset...")
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")
    print(f"  Shape: {df.shape}")

    # ── Filter only PM25 and drop NaN ─────────────────────────────────────────
    print("\nPreparing PM2.5 data...")
    pm25 = df[["location_id", "datetime", "PM25"]].dropna(subset=["PM25"])
    pm25["datetime"] = pd.to_datetime(pm25["datetime"])

    # ── Create violation flag ─────────────────────────────────────────────────
    pm25["violation"] = (pm25["PM25"] > THRESHOLD).astype(int)

    # ── Aggregate: daily violation rate per location ──────────────────────────
    print("Aggregating to daily violation rate per location...")
    pm25["date"] = pm25["datetime"].dt.date
    daily = (pm25.groupby(["location_id", "date"])["violation"]
                 .mean()
                 .reset_index())
    daily.columns = ["location_id", "date", "violation_rate"]
    daily["date"] = pd.to_datetime(daily["date"])

    # ── Pivot to matrix: rows = locations, columns = dates ────────────────────
    matrix = daily.pivot_table(
        index="location_id",
        columns="date",
        values="violation_rate"
    ).fillna(0)

    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Locations: {matrix.shape[0]}, Days: {matrix.shape[1]}")

    # ── Sort locations by total violation rate (worst at top) ─────────────────
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    print("\nPlotting heatmap...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [4, 1]})
    fig.patch.set_facecolor("white")

    # ── Panel 1: Heatmap ──────────────────────────────────────────────────────
    ax1 = axes[0]
    im = ax1.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="nearest"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)
    cbar.set_label("Hourly Violation Rate\n(fraction of hours PM2.5 > 35 µg/m³)",
                   fontsize=9)

    # X axis — show month labels
    dates = matrix.columns
    month_positions = []
    month_labels = []
    current_month = None
    for i, d in enumerate(dates):
        if d.month != current_month:
            month_positions.append(i)
            month_labels.append(d.strftime("%b %Y"))
            current_month = d.month

    ax1.set_xticks(month_positions)
    ax1.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=9)

    # Y axis — location IDs
    ax1.set_yticks(range(len(matrix.index)))
    ax1.set_yticklabels([f"Loc {lid}" for lid in matrix.index], fontsize=7)
    ax1.set_ylabel("Sensor Location", fontsize=11)
    ax1.set_title(
        f"PM2.5 Health Threshold Violations Across {matrix.shape[0]} Sensors Over Time\n"
        f"(darker = more hours exceeding {THRESHOLD} µg/m³)",
        fontsize=12, fontweight="bold"
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: Daily average violation rate (all sensors combined) ──────────
    ax2 = axes[1]
    daily_avg = matrix.mean(axis=0)

    ax2.fill_between(range(len(daily_avg)), daily_avg.values,
                     color="#E63946", alpha=0.6)
    ax2.plot(range(len(daily_avg)), daily_avg.values,
             color="#E63946", linewidth=0.8)
    ax2.set_xticks(month_positions)
    ax2.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Avg Violation\nRate", fontsize=9)
    ax2.set_title("Network-Wide Daily Violation Rate", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, len(daily_avg) - 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "task2_temporal.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot to {out_path}")

    # ── Periodic signature analysis ───────────────────────────────────────────
    print("\n── Periodic Signature Analysis ──────────────────────────────────")

    # Hour of day pattern
    pm25["hour"] = pm25["datetime"].dt.hour
    hourly_pattern = pm25.groupby("hour")["violation"].mean()
    peak_hour = hourly_pattern.idxmax()
    print(f"  Peak violation hour: {peak_hour}:00 (rate: {hourly_pattern[peak_hour]:.3f})")

    # Day of week pattern
    pm25["dayofweek"] = pm25["datetime"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_pattern = pm25.groupby("dayofweek")["violation"].mean().reindex(dow_order)
    peak_day = dow_pattern.idxmax()
    print(f"  Peak violation day: {peak_day} (rate: {dow_pattern[peak_day]:.3f})")

    # Monthly pattern
    pm25["month"] = pm25["datetime"].dt.month
    monthly_pattern = pm25.groupby("month")["violation"].mean()
    peak_month = monthly_pattern.idxmax()
    print(f"  Peak violation month: {peak_month} (rate: {monthly_pattern[peak_month]:.3f})")

    print("\n  Conclusion:")
    print(f"  Violations peak at hour {peak_hour}:00 → suggests daily traffic cycle driver")
    print(f"  Monthly variation also present → seasonal shifts contribute too")

if __name__ == "__main__":
    main()