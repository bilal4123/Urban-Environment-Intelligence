import requests
import pandas as pd

API_KEY = "99b5060aa154232969307114212de8f26151716f865f1abf9602abd07b25a14b"
HEADERS = {"X-API-Key": API_KEY}
BASE_API = "https://api.openaq.org/v3"

def get_locations():
    all_locations = []
    page = 1
    while len(all_locations) < 100:
        params = {
            "parameters_id": 2,  # 2 = PM2.5
            "limit": 100,
            "page": page
        }
        r = requests.get(f"{BASE_API}/locations", headers=HEADERS, params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        all_locations.extend(results)
        page += 1

    # Keep only first 100
    all_locations = all_locations[:100]

    # Extract useful fields
    rows = []
    for loc in all_locations:
        rows.append({
            "location_id": loc["id"],
            "name": loc.get("name", ""),
            "city": loc.get("locality", ""),
            "country": loc.get("country", {}).get("code", ""),
            "latitude": loc.get("coordinates", {}).get("latitude", None),
            "longitude": loc.get("coordinates", {}).get("longitude", None),
        })

    df = pd.DataFrame(rows)
    df.to_csv("locations_100.csv", index=False)
    print(f"Saved {len(df)} locations to locations_100.csv")
    print(df.head(10))
    return df

if __name__ == "__main__":
    get_locations()