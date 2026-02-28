import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # ── Load all datasets ─────────────────────────────────────────────────────
    print("Loading datasets...")
    locations  = pd.read_csv(RAW_DIR / "locations_100.csv")
    pollutants = pd.read_parquet(RAW_DIR / "measurements_openaq.parquet")
    weather    = pd.read_parquet(RAW_DIR / "weather_openmeteo.parquet")

    print(f"  Locations:  {locations.shape}")
    print(f"  Pollutants: {pollutants.shape}")
    print(f"  Weather:    {weather.shape}")

    # ── Clean pollutants ──────────────────────────────────────────────────────
    print("\nCleaning pollutants...")
    pollutants["datetime"] = pd.to_datetime(pollutants["datetime"], utc=True)
    pollutants["datetime"] = pollutants["datetime"].dt.floor("h")
    pollutants["datetime"] = pollutants["datetime"].dt.tz_localize(None)
    pollutants = pollutants.dropna(subset=["value", "datetime"])
    pollutants = pollutants[pollutants["value"] >= 0]

    # Average duplicates
    pollutants = (pollutants
                  .groupby(["location_id", "parameter", "datetime"])["value"]
                  .mean()
                  .reset_index())

    # Pivot wide
    pollutants_wide = pollutants.pivot_table(
        index=["location_id", "datetime"],
        columns="parameter",
        values="value"
    ).reset_index()
    pollutants_wide.columns.name = None
    print(f"  Pollutants wide shape: {pollutants_wide.shape}")
    print(f"  Unique locations: {pollutants_wide['location_id'].nunique()}")

    # ── Clean weather ─────────────────────────────────────────────────────────
    print("\nCleaning weather...")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["datetime"] = weather["datetime"].dt.tz_localize(None)
    weather = weather.dropna(subset=["temperature", "humidity"])

    # ── Merge pollutants + weather (left join keeps all pollutant rows) ────────
    print("\nMerging pollutants and weather...")
    df = pd.merge(pollutants_wide, weather, on=["location_id", "datetime"], how="left")
    print(f"  Merged shape: {df.shape}")
    print(f"  Rows WITH weather data: {df['temperature'].notna().sum()}")
    print(f"  Rows WITHOUT weather data: {df['temperature'].isna().sum()}")

    # ── Fill missing temp/humidity with per-location average ──────────────────
    print("\nFilling missing temperature & humidity with location averages...")
    loc_avg = df.groupby("location_id")[["temperature", "humidity"]].transform("mean")
    df["temperature"] = df["temperature"].fillna(loc_avg["temperature"])
    df["humidity"]    = df["humidity"].fillna(loc_avg["humidity"])

    # If still missing (location had no weather overlap), fill with global average
    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())
    df["humidity"]    = df["humidity"].fillna(df["humidity"].mean())

    print(f"  Remaining missing temperature: {df['temperature'].isna().sum()}")
    print(f"  Remaining missing humidity: {df['humidity'].isna().sum()}")

    # ── Add location metadata ─────────────────────────────────────────────────
    print("\nAdding location metadata...")
    df = pd.merge(df, locations, on="location_id", how="left")

    # ── Add zone column ───────────────────────────────────────────────────────
    print("\nAssigning zones...")
    no2_median = df.groupby("location_id")["no2"].median()
    industrial_ids = no2_median[no2_median >= no2_median.median()].index
    df["zone"] = df["location_id"].apply(
        lambda x: "Industrial" if x in industrial_ids else "Residential"
    )
    print(f"  Industrial: {df[df['zone']=='Industrial']['location_id'].nunique()} locations")
    print(f"  Residential: {df[df['zone']=='Residential']['location_id'].nunique()} locations")

    # ── Final cleanup ─────────────────────────────────────────────────────────
    print("\nFinal cleanup...")
    df = df.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    df = df.rename(columns={
        "pm25": "PM25",
        "pm10": "PM10",
        "no2":  "NO2",
        "o3":   "Ozone",
        "temperature": "Temperature",
        "humidity": "Humidity"
    })

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique locations: {df['location_id'].nunique()}")
    print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"\nSample:\n{df.head()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "final_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✅ Saved final dataset to {out_path}")

if __name__ == "__main__":
    main()