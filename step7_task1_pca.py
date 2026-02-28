import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["PM25", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]
COLORS   = {"Industrial": "#E63946", "Residential": "#457B9D"}

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading final dataset...")
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")
    print(f"  Shape: {df.shape}")

    # ── Aggregate to one row per location (median of all readings) ────────────
    print("\nAggregating to location level...")
    location_df = (df.groupby(["location_id", "zone"])[FEATURES]
                     .median()
                     .reset_index())
    print(f"  Locations: {len(location_df)}")
    print(f"  Zone counts:\n{location_df['zone'].value_counts()}")

    # Drop rows with any NaN in features
    location_df = location_df.dropna(subset=FEATURES)
    print(f"  Locations after dropping NaN: {len(location_df)}")
    # ── Standardize ───────────────────────────────────────────────────────────
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(location_df[FEATURES])

    # ── Apply PCA ─────────────────────────────────────────────────────────────
    print("\nApplying PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_ * 100
    print(f"  PC1 explains: {explained[0]:.1f}%")
    print(f"  PC2 explains: {explained[1]:.1f}%")
    print(f"  Total explained: {sum(explained):.1f}%")

    # ── Loadings ──────────────────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1", "PC2"]
    )
    print(f"\nPCA Loadings:\n{loadings.round(3)}")

    # ── Build plot dataframe ───────────────────────────────────────────────────
    plot_df = location_df[["location_id", "zone"]].copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]

    # ── Figure: 2 panels (scatter + loadings bar chart) ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # ── Panel 1: PCA Scatter ──────────────────────────────────────────────────
    ax1 = axes[0]
    for zone, color in COLORS.items():
        mask = plot_df["zone"] == zone
        ax1.scatter(
            plot_df.loc[mask, "PC1"],
            plot_df.loc[mask, "PC2"],
            c=color,
            label=zone,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            s=100
        )

    ax1.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=11)
    ax1.set_title("PCA Projection: Industrial vs Residential Zones", fontsize=12, fontweight="bold")
    ax1.legend(title="Zone", frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    # ── Panel 2: Loadings Bar Chart ───────────────────────────────────────────
    ax2 = axes[1]
    x = np.arange(len(FEATURES))
    width = 0.35

    bars1 = ax2.bar(x - width/2, loadings["PC1"], width,
                    label="PC1", color="#E63946", alpha=0.85)
    bars2 = ax2.bar(x + width/2, loadings["PC2"], width,
                    label="PC2", color="#457B9D", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(FEATURES, rotation=15, ha="right")
    ax2.set_ylabel("Loading Score", fontsize=11)
    ax2.set_title("PCA Loadings: Variable Contributions", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "task1_pca.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved plot to {out_path}")

    # ── Print analysis summary ────────────────────────────────────────────────
    print("\n── Analysis Summary ─────────────────────────────────────────────")
    pc1_driver = loadings["PC1"].abs().idxmax()
    pc2_driver = loadings["PC2"].abs().idxmax()
    print(f"  Main driver of PC1: {pc1_driver} (loading: {loadings.loc[pc1_driver, 'PC1']:.3f})")
    print(f"  Main driver of PC2: {pc2_driver} (loading: {loadings.loc[pc2_driver, 'PC2']:.3f})")
    print(f"  PC1 separates zones primarily based on pollution intensity.")
    print(f"  PC2 captures weather/environmental variation.")

if __name__ == "__main__":
    main()