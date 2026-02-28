# Urban Environmental Intelligence — Assignment 02

## Overview
A diagnostic engine for identifying environmental anomalies across 
100 global air quality sensor nodes. Built using real data from 
OpenAQ and Open-Meteo APIs.

## Data Sources
- **Pollutants** (PM2.5, PM10, NO2, Ozone): OpenAQ Global Air Quality API
- **Weather** (Temperature, Humidity): Open-Meteo Historical Archive API

## Project Structure
```
data-science-assignment-2/
├── data/
│   ├── raw/                        # Raw downloaded data
│   │   ├── locations_100.csv
│   │   ├── sensors_100.csv
│   │   ├── measurements_openaq.parquet
│   │   └── weather_openmeteo.parquet
│   └── processed/
│       └── final_dataset.parquet   # Final merged dataset
├── scripts/
│   ├── step2_get_locations.py      # Fetch 100 station IDs
│   ├── step3_get_sensors.py        # Fetch sensor IDs
│   ├── step4_get_measurements.py   # Download pollutant data
│   ├── step5_get_weather.py        # Download weather data
│   ├── step6_merge_data.py         # Merge all datasets
│   ├── step7_task1_pca.py          # Task 1: PCA analysis
│   ├── step8_task2_temporal.py     # Task 2: Temporal heatmap
│   ├── step9_task3_distribution.py # Task 3: Distribution plots
│   └── step10_task4_visual_integrity.py # Task 4: Visual audit
├── outputs/                        # All generated plots
│   ├── task1_pca.png
│   ├── task2_temporal.png
│   ├── task3_distribution.png
│   └── task4_visual_integrity.png
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd data-science-assignment-2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your OpenAQ API key
- Register at https://explore.openaq.org
- Copy your API key
- Replace `paste-your-api-key-here` in scripts 2, 3, 4

## How to Run

Run scripts in order:
```bash
python scripts/step2_get_locations.py
python scripts/step3_get_sensors.py
python scripts/step4_get_measurements.py
python scripts/step5_get_weather.py
python scripts/step6_merge_data.py
python scripts/step7_task1_pca.py
python scripts/step8_task2_temporal.py
python scripts/step9_task3_distribution.py
python scripts/step10_task4_visual_integrity.py
```

Then launch the dashboard:
```bash
python -m streamlit run dashboard/app.py
```

## Task Summary

### Task 1 — Dimensionality Reduction (PCA)
- Standardized 6 variables and applied PCA
- PC1 (55.8% variance): driven by PM10 & PM25 — separates Industrial from Residential
- PC2 (18.3% variance): driven by Ozone & Temperature — captures climate variation

### Task 2 — High-Density Temporal Analysis
- Heatmap of PM2.5 violations across 67 sensors over time
- Violations peak at 2AM (daily traffic cycle) and in winter months (seasonal)

### Task 3 — Distribution Modeling & Tail Integrity
- KDE plot reveals bulk distribution peaks
- CCDF (log-log) reveals long tail — P(PM2.5 > 200) = 2.075%
- 99th percentile = 273 µg/m³ in industrial zones

### Task 4 — Visual Integrity Audit
- Rejected 3D bar chart (Lie Factor > 1.0, poor Data-Ink Ratio)
- Implemented Small Multiples with sequential YlOrRd colormap
- Sequential colormap chosen for perceptually uniform luminance

## Key Findings
- Indian industrial sensors (I-103, I-17) show PM2.5 up to 121 µg/m³
- 2.075% of industrial zone hours exceed extreme hazard threshold (200 µg/m³)
- Pollution violations follow both daily (24h) and seasonal (monthly) patterns
- PCA explains 74% of variance in just 2 dimensions