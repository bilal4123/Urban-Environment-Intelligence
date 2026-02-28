import requests
import pandas as pd
import time
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
DATE_FROM = "2024-12-01"
DATE_TO = "2025-02-28"

def fetch_weather(location_id, lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": DATE_FROM,
        "end_date": DATE_TO,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "UTC"
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame({
                "datetime": data["hourly"]["time"],
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
            })
            df["location_id"] = location_id
            return df
    except Exception as e:
        print(f"  Error location {location_id}: {e}")
    return pd.DataFrame()

def main():
    locations = pd.read_csv(RAW_DIR / "locations_100.csv")
    print(f"Fetching weather for {len(locations)} locations...")

    all_frames = []
    for i, row in locations.iterrows():
        loc_id = row["location_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        print(f"[{i+1}/100] Location {loc_id} ({lat}, {lon})...", end=" ", flush=True)

        df = fetch_weather(loc_id, lat, lon)
        if not df.empty:
            all_frames.append(df)
            print(f"✓ {len(df)} records")
        else:
            print("✗ no data")

        time.sleep(0.5)  # be polite to the API

    if all_frames:
        final = pd.concat(all_frames, ignore_index=True)
        out_path = RAW_DIR / "weather_openmeteo.parquet"
        final.to_parquet(out_path, index=False)
        print(f"\nDone! Saved {len(final)} records to {out_path}")
        print(final.head(5))
    else:
        print("\nNo weather data fetched!")

if __name__ == "__main__":
    main()