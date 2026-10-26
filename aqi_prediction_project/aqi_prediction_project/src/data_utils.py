from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
REQUIRED_COLUMNS = REQUIRED_FEATURES + ["AQI"]

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("CSV has no rows.")
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
    if df.empty:
        raise ValueError("No valid numeric rows remain after cleaning.")
    if (df[REQUIRED_FEATURES] < 0).any().any():
        raise ValueError("Pollutant values must be non-negative.")
    return df

def load_dataset(path):
    df = pd.read_csv(path)
    return validate_columns(df)

def make_demo_dataset(n=1200, seed=42):
    """Create synthetic demonstration data; it is not measured environmental data."""
    rng = np.random.default_rng(seed)
    pm25 = rng.gamma(2.1, 18, n).clip(1, 250)
    pm10 = (pm25 * rng.uniform(1.2, 2.1, n) + rng.normal(0, 10, n)).clip(2, 450)
    no2 = (rng.gamma(2, 12, n) + pm25 * 0.12).clip(1, 160)
    so2 = (rng.gamma(1.8, 5, n) + pm25 * 0.04).clip(0.1, 100)
    co = (rng.gamma(1.7, 0.35, n) + pm25 * 0.003).clip(0.05, 8)
    o3 = (rng.gamma(2, 14, n) + rng.normal(0, 5, n)).clip(1, 180)
    # Synthetic target relationship for demonstrating a regression workflow only.
    aqi = (0.72 * pm25 + 0.18 * pm10 + 0.22 * no2 + 0.15 * so2 + 5.0 * co + 0.12 * o3
           + rng.normal(0, 8, n)).clip(0, 500)
    df = pd.DataFrame({"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3, "AQI": aqi})
    return df
