import requests
import pandas as pd
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
API_KEY = "99b5060aa154232969307114212de8f26151716f865f1abf9602abd07b25a14b"
HEADERS = {"X-API-Key": API_KEY}
BASE_API = "https://api.openaq.org/v3"
DATE_FROM = "2024-12-01T00:00:00Z"  # ~90 days back
DATE_TO   = "2025-02-28T23:59:59Z"
PARAMETERS = ["pm25", "pm10", "no2", "o3"]

def fetch_sensor(row):
    sensor_id  = row["sensor_id"]
    location_id = row["location_id"]
    parameter  = row["parameter"]
    all_data   = []
    page = 1
    while True:
        params = {
            "date_from": DATE_FROM,
            "date_to":   DATE_TO,
            "limit": 1000,
            "page": page
        }
        try:
            r = requests.get(
                f"{BASE_API}/sensors/{sensor_id}/hours",
                headers=HEADERS,
                params=params,
                timeout=30
            )
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
            if not results:
                break
            for entry in results:
                all_data.append({
                    "location_id": location_id,
                    "sensor_id":   sensor_id,
                    "parameter":   parameter,
                    "datetime":    entry.get("period", {}).get("datetimeFrom", {}).get("utc"),
                    "value":       entry.get("value"),
                })
            page += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  Error sensor {sensor_id}: {e}")
            break
    return all_data

def main():
    sensors = pd.read_csv(RAW_DIR / "sensors_100.csv")
    sensors = sensors[sensors["parameter"].isin(PARAMETERS)]
    print(f"Fetching {len(sensors)} sensors with 4 parallel workers...")

    all_rows = []
    done = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_sensor, row): row for _, row in sensors.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            all_rows.extend(result)
            done += 1
            row = futures[future]
            print(f"[{done}/{len(sensors)}] sensor {row['sensor_id']} ({row['parameter']}) → {len(result)} records")

            # checkpoint every 50 sensors
            if done % 50 == 0:
                pd.DataFrame(all_rows).to_parquet(RAW_DIR / "measurements_checkpoint.parquet", index=False)
                print(f"  >>> Checkpoint saved at {done} sensors")

    df = pd.DataFrame(all_rows)
    df.to_parquet(RAW_DIR / "measurements_openaq.parquet", index=False)
    print(f"\nDone! Saved {len(df)} records to measurements_openaq.parquet")
    print(df["parameter"].value_counts())

if __name__ == "__main__":
    main()