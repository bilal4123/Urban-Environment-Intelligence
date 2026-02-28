import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
FEATURES = ["PM25", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]
THRESHOLD = 35
EXTREME_THRESHOLD = 200

st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 Urban Environmental Intelligence")
st.sidebar.markdown("**Data Source:** OpenAQ + Open-Meteo")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Task 1: PCA Analysis", "Task 2: Temporal Analysis",
     "Task 3: Distribution", "Task 4: Visual Integrity"]
)

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("🌍 Urban Environmental Intelligence Dashboard")
    st.markdown("### Smart City Air Quality Diagnostic Engine")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Sensor Locations", df["location_id"].nunique())
    col3.metric("Countries", df["country"].nunique())
    col4.metric("Date Range", f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zone Distribution")
        zone_counts = df.groupby("zone")["location_id"].nunique().reset_index()
        fig = px.pie(zone_counts, values="location_id", names="zone",
                     color_discrete_map={"Industrial": "#E63946", "Residential": "#457B9D"})
        fig.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("PM2.5 by Country")
        country_pm25 = (df.groupby("country")["PM25"]
                          .median()
                          .sort_values(ascending=False)
                          .reset_index())
        fig = px.bar(country_pm25, x="country", y="PM25",
                     color="PM25", color_continuous_scale="YlOrRd",
                     labels={"PM25": "Median PM2.5 (µg/m³)"})
        fig.add_hline(y=35, line_dash="dash", line_color="red",
                      annotation_text="Health Threshold")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(100), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: PCA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Task 1: PCA Analysis":
    st.title("Task 1: Dimensionality Reduction (PCA)")
    st.markdown("Projecting 6 environmental variables into 2D to reveal zone clusters.")
    st.markdown("---")

    location_df = (df.groupby(["location_id", "zone"])[FEATURES]
                     .median()
                     .reset_index()
                     .dropna(subset=FEATURES))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(location_df[FEATURES])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    location_df["PC1"] = X_pca[:, 0]
    location_df["PC2"] = X_pca[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"PCA Scatter (Total variance: {sum(explained):.1f}%)")
        fig = px.scatter(
            location_df, x="PC1", y="PC2", color="zone",
            color_discrete_map={"Industrial": "#E63946", "Residential": "#457B9D"},
            hover_data=["location_id"],
            labels={
                "PC1": f"PC1 ({explained[0]:.1f}% variance)",
                "PC2": f"PC2 ({explained[1]:.1f}% variance)"
            }
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("PCA Loadings")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=FEATURES,
            columns=["PC1", "PC2"]
        ).reset_index()
        loadings.columns = ["Feature", "PC1", "PC2"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="PC1", x=loadings["Feature"],
                             y=loadings["PC1"], marker_color="#E63946"))
        fig.add_trace(go.Bar(name="PC2", x=loadings["Feature"],
                             y=loadings["PC2"], marker_color="#457B9D"))
        fig.update_layout(barmode="group", height=450,
                          yaxis_title="Loading Score")
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"**PC1 ({explained[0]:.1f}%)** is driven by PM10 & PM25 — separates Industrial from Residential zones. "
            f"**PC2 ({explained[1]:.1f}%)** is driven by Ozone & Temperature — captures climate variation.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Task 2: Temporal Analysis":
    st.title("Task 2: High-Density Temporal Analysis")
    st.markdown("PM2.5 health threshold violations across all sensors over time.")
    st.markdown("---")

    pm25 = df[["location_id", "datetime", "PM25"]].dropna(subset=["PM25"])
    pm25 = pm25[pm25["PM25"] > 0].copy()
    pm25["violation"] = (pm25["PM25"] > THRESHOLD).astype(int)
    pm25["date"] = pm25["datetime"].dt.date

    daily = (pm25.groupby(["location_id", "date"])["violation"]
                 .mean()
                 .reset_index())
    daily.columns = ["location_id", "date", "violation_rate"]
    daily["date"] = pd.to_datetime(daily["date"])

    matrix = daily.pivot_table(
        index="location_id", columns="date", values="violation_rate"
    ).fillna(0)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    st.subheader("PM2.5 Violation Heatmap (All Sensors)")
    fig = px.imshow(
        matrix,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels={"color": "Violation Rate"},
        title=f"Fraction of hours PM2.5 > {THRESHOLD} µg/m³ per day per sensor"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hourly Pattern (24h cycle)")
        pm25["hour"] = pm25["datetime"].dt.hour
        hourly = pm25.groupby("hour")["violation"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="violation",
                      labels={"violation": "Avg Violation Rate", "hour": "Hour of Day"},
                      markers=True)
        fig.update_traces(line_color="#E63946")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Pattern (seasonal)")
        pm25["month"] = pm25["datetime"].dt.month
        monthly = pm25.groupby("month")["violation"].mean().reset_index()
        fig = px.bar(monthly, x="month", y="violation",
                     labels={"violation": "Avg Violation Rate", "month": "Month"},
                     color="violation", color_continuous_scale="YlOrRd")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Task 3: Distribution":
    st.title("Task 3: Distribution Modeling & Tail Integrity")
    st.markdown("Probability of extreme PM2.5 hazard events in industrial zones.")
    st.markdown("---")

    industrial = df[df["zone"] == "Industrial"]["PM25"].dropna()
    industrial = industrial[industrial > 0]
    p99 = np.percentile(industrial, 99)
    prob_extreme = (industrial > EXTREME_THRESHOLD).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("99th Percentile", f"{p99:.1f} µg/m³")
    col2.metric("P(PM2.5 > 200)", f"{prob_extreme:.4f}")
    col3.metric("Extreme Records", f"{(industrial > EXTREME_THRESHOLD).sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KDE Plot — Reveals Peaks")
        peak_data = industrial[industrial <= 300]
        kde = stats.gaussian_kde(peak_data, bw_method=0.1)
        x_range = np.linspace(0, 300, 500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range),
                                 fill="tozeroy", line_color="#457B9D",
                                 name="KDE"))
        fig.add_vline(x=35, line_dash="dash", line_color="#E63946",
                      annotation_text="Health Threshold")
        fig.add_vline(x=p99, line_dash="dash", line_color="#F4A261",
                      annotation_text=f"99th pct ({p99:.0f})")
        fig.update_layout(height=400,
                          xaxis_title="PM2.5 (µg/m³)",
                          yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("CCDF — Reveals Tails (Log Scale)")
        sorted_data = np.sort(industrial.values)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        sample_idx = np.unique(np.logspace(0, np.log10(len(sorted_data)-1),
                                           2000).astype(int))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_data[sample_idx],
                                 y=ccdf[sample_idx],
                                 line_color="#E63946", name="CCDF"))
        fig.add_vline(x=EXTREME_THRESHOLD, line_dash="dash",
                      line_color="#2D6A4F",
                      annotation_text=f"Extreme Hazard ({EXTREME_THRESHOLD})")
        fig.update_layout(height=400,
                          xaxis_type="log", yaxis_type="log",
                          xaxis_title="PM2.5 (µg/m³) Log Scale",
                          yaxis_title="P(X > x) Log Scale")
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Most honest plot for rare events: CCDF (log-log scale).** "
            f"The KDE compresses the tail making extreme events invisible. "
            f"The CCDF stretches the tail revealing that {prob_extreme:.2%} "
            f"of industrial hours exceed the extreme hazard threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: VISUAL INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Task 4: Visual Integrity":
    st.title("Task 4: Visual Integrity Audit")
    st.markdown("Rejecting the 3D bar chart proposal and implementing a better solution.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.error("❌ **REJECTED: 3D Bar Chart**")
        st.markdown("""
        **Lie Factor Violations:**
        - 3D depth distorts bar heights — front bars appear larger than back bars
        - Perspective projection makes equal values look unequal
        - Lie Factor > 1.0 (Tufte's principle violated)

        **Data-Ink Ratio Violations:**
        - Shadows, depth effects, perspective grids add zero information
        - Every non-data pixel wastes the viewer's attention
        - Occlusion hides data behind other bars
        """)

    with col2:
        st.success("✅ **ACCEPTED: Small Multiples**")
        st.markdown("""
        **Why Small Multiples works:**
        - Bar length → activity/density (Variable 1)
        - Color (YlOrRd sequential) → PM2.5 severity (Variable 2)
        - Panel position → Region/Country (Variable 3)
        - Zero distortion, zero occlusion, zero false perception

        **Color Scale Justification (Sequential over Rainbow):**
        - Sequential maps luminance monotonically to values
        - Human perception of luminance is linear
        - Rainbow (jet) has non-uniform luminance jumps → false boundaries
        - YlOrRd is perceptually honest and colorblind-friendly
        """)

    st.markdown("---")
    st.subheader("Interactive Small Multiples: Pollution vs Activity vs Region")

    location_df = (df.groupby(["location_id", "zone", "country"])
                     .agg(PM25=("PM25", "median"),
                          activity=("location_id", "count"))
                     .reset_index())
    location_df["activity"] = (location_df["activity"] /
                                location_df["activity"].max() * 1000).round(0)
    location_df["label"] = (location_df["zone"].str[0] + "-" +
                             location_df["location_id"].astype(str))

    # Get top 8 countries
    top_countries = (location_df["country"]
                     .value_counts()
                     .head(8)
                     .index.tolist())
    plot_df = location_df[location_df["country"].isin(top_countries)].copy()

    # One subplot per country
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    ncols = 4
    nrows = int(np.ceil(len(top_countries) / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=top_countries,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    colorscale = px.colors.sequential.YlOrRd
    pm25_min = plot_df["PM25"].min()
    pm25_max = plot_df["PM25"].max()

    def get_color(value):
        ratio = (value - pm25_min) / (pm25_max - pm25_min + 1e-9)
        idx = int(ratio * (len(colorscale) - 1))
        return colorscale[min(idx, len(colorscale) - 1)]

    for i, country in enumerate(top_countries):
        row = i // ncols + 1
        col = i % ncols + 1
        country_df = plot_df[plot_df["country"] == country].copy()
        country_df = country_df.sort_values("activity")

        fig.add_trace(
            go.Bar(
                x=country_df["activity"],
                y=country_df["label"],
                orientation="h",
                marker=dict(
                    color=country_df["PM25"],
                    colorscale="YlOrRd",
                    cmin=pm25_min,
                    cmax=pm25_max,
                    showscale=(i == 0)
                ),
                text=country_df["PM25"].round(0).astype(int),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Activity: %{x}<br>"
                    "PM2.5: %{text} µg/m³<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=350 * nrows,
        title=dict(
            text="Pollution vs Activity Level vs Region<br>"
                 "<sup>I=Industrial, R=Residential | Color=PM2.5 Severity</sup>",
            font=dict(size=14)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=9),
        coloraxis=dict(
            colorscale="YlOrRd",
            colorbar=dict(title="PM2.5<br>(µg/m³)")
        )
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)