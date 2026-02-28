import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading final dataset...")
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")

    # ── Aggregate to location level ───────────────────────────────────────────
    print("Aggregating to location level...")
    location_df = (df.groupby(["location_id", "zone", "country"])
                     .agg(
                         PM25=("PM25", "median"),
                         population_density=("location_id", "count")  # proxy: data density
                     )
                     .reset_index())

    # Normalize population density to a readable scale
    location_df["population_density"] = (
        location_df["population_density"] /
        location_df["population_density"].max() * 1000
    ).round(0)

    print(f"  Locations: {len(location_df)}")
    print(f"  Countries: {location_df['country'].nunique()}")

    # ── Get top countries by number of locations ───────────────────────────────
    top_countries = (location_df["country"]
                     .value_counts()
                     .head(8)
                     .index.tolist())
    plot_df = location_df[location_df["country"].isin(top_countries)].copy()

    # ── Figure: Small Multiples (one panel per country) ───────────────────────
    print("\nPlotting small multiples...")
    n_countries = len(top_countries)
    ncols = 4
    nrows = int(np.ceil(n_countries / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(16, 5 * nrows),
                             sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    # Sequential colormap — perceptually uniform (no rainbow)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(
        vmin=plot_df["PM25"].min(),
        vmax=plot_df["PM25"].max()
    )

    for i, country in enumerate(top_countries):
        ax = axes[i]
        country_df = plot_df[plot_df["country"] == country].copy()
        country_df = country_df.sort_values("population_density")

        colors = [cmap(norm(v)) for v in country_df["PM25"]]

        bars = ax.barh(
            range(len(country_df)),
            country_df["population_density"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            height=0.7
        )

        # Zone labels on y axis
        ax.set_yticks(range(len(country_df)))
        ax.set_yticklabels(
            [f"{'I' if z == 'Industrial' else 'R'}-{lid}"
             for z, lid in zip(country_df["zone"], country_df["location_id"])],
            fontsize=8
        )

        ax.set_title(f"{country}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Data Density (proxy for activity)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate PM2.5 value on each bar
        for j, (_, row) in enumerate(country_df.iterrows()):
            ax.text(
                row["population_density"] + 5, j,
                f"{row['PM25']:.0f}",
                va="center", fontsize=7, color="black"
            )

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_countries],
                        fraction=0.02, pad=0.04,
                        location="right")
    cbar.set_label("Median PM2.5 (µg/m³)", fontsize=10)

    fig.suptitle(
        "Pollution vs Activity Level vs Region\n"
        "Small Multiples — I=Industrial, R=Residential | Color=PM2.5 Severity",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "task4_visual_integrity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot to {out_path}")

    # ── Print audit justification ─────────────────────────────────────────────
    print("\n── Visual Integrity Audit ───────────────────────────────────────")
    print("""
  DECISION: REJECT the 3D bar chart proposal.

  REASON 1 — Lie Factor:
  3D bars create false depth perception. The front face of a 3D bar
  appears larger than the back face even if values are equal,
  introducing a Lie Factor > 1.0 (Tufte's principle).

  REASON 2 — Data-Ink Ratio:
  3D effects (shadows, depth, perspective grid) add non-data ink
  with zero informational value, violating Tufte's data-ink ratio
  principle. Every pixel should encode data.

  SOLUTION: Small Multiples approach.
  - One panel per region (country) for clean comparison
  - Bar length encodes activity/density (1st variable)
  - Color encodes PM2.5 severity (2nd variable)  
  - Panel position encodes region (3rd variable)
  - All three variables readable simultaneously with zero distortion

  COLOR SCALE JUSTIFICATION: Sequential (YlOrRd) over Rainbow.
  - Sequential scales map luminance monotonically to data values
  - Human perception of luminance is linear — light=low, dark=high
  - Rainbow (jet) colormap has non-uniform luminance jumps that
    create false boundaries and mislead the viewer about data values
  - YlOrRd is perceptually honest and colorblind-friendly
    """)

if __name__ == "__main__":
    main()