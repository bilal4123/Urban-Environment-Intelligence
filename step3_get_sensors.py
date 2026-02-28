import requests
import pandas as pd
import time

API_KEY = "99b5060aa154232969307114212de8f26151716f865f1abf9602abd07b25a14b"
HEADERS = {"X-API-Key": API_KEY}
BASE_API = "https://api.openaq.org/v3"

# Parameters we care about (OpenAQ parameter IDs)
PARAMETER_NAMES = ["pm25", "pm10", "no2", "o3", "temperature", "humidity"]

def get_sensors_for_location(location_id):
    r = requests.get(f"{BASE_API}/locations/{location_id}/sensors", headers=HEADERS)
    if r.status_code != 200:
        return []
    return r.json().get("results", [])

def main():
    locations = pd.read_csv("locations_100.csv")
    all_sensors = []

    for i, row in locations.iterrows():
        loc_id = row["location_id"]
        print(f"[{i+1}/100] Fetching sensors for location {loc_id}...")
        sensors = get_sensors_for_location(loc_id)

        for s in sensors:
            param = s.get("parameter", {})
            param_name = param.get("name", "").lower()
            if param_name in PARAMETER_NAMES:
                all_sensors.append({
                    "location_id": loc_id,
                    "sensor_id": s["id"],
                    "parameter": param_name,
                    "unit": param.get("units", ""),
                })
        time.sleep(0.3)  # be polite to the API

    df = pd.DataFrame(all_sensors)
    df.to_csv("sensors_100.csv", index=False)
    print(f"\nDone! Saved {len(df)} sensors to sensors_100.csv")
    print(df["parameter"].value_counts())

if __name__ == "__main__":
    main()