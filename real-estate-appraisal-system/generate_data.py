"""
generate_data.py
-----------------
Generates a synthetic but realistic real estate dataset for the
Real Estate Appraisal System project and saves it to data/housing_data.csv.

This script is provided for transparency/reproducibility. The CSV file
it produces is already included in the repository under data/.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500

# --- Core structural features -------------------------------------------------
square_footage = np.random.normal(2000, 700, N).clip(500, 6000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
bathrooms = np.round(np.clip(bedrooms * 0.75 + np.random.normal(0, 0.6, N), 1, 6) * 2) / 2
lot_size = np.random.normal(8000, 3500, N).clip(1000, 30000)
year_built = np.random.randint(1950, 2024, N)
house_age = 2024 - year_built

# --- Location / neighborhood quality ------------------------------------------
neighborhood_quality = np.random.choice(
    ["Low", "Medium", "High", "Premium"], N, p=[0.2, 0.35, 0.30, 0.15]
)
neighborhood_multiplier = pd.Series(neighborhood_quality).map(
    {"Low": 0.75, "Medium": 1.0, "High": 1.3, "Premium": 1.7}
).values

distance_to_city_center = np.random.exponential(8, N).clip(0.5, 50)

# --- Amenities ------------------------------------------------------------------
garage_spaces = np.random.choice([0, 1, 2, 3], N, p=[0.15, 0.35, 0.4, 0.10])
has_pool = np.random.choice([0, 1], N, p=[0.85, 0.15])
has_garden = np.random.choice([0, 1], N, p=[0.4, 0.6])
school_rating = np.random.randint(1, 11, N)  # 1-10
crime_rate = np.random.exponential(3, N).clip(0, 20)  # incidents per 1000 residents
renovated = np.random.choice([0, 1], N, p=[0.7, 0.3])

condition = np.random.choice(
    ["Poor", "Fair", "Good", "Excellent"], N, p=[0.05, 0.25, 0.45, 0.25]
)
condition_multiplier = pd.Series(condition).map(
    {"Poor": 0.7, "Fair": 0.88, "Good": 1.0, "Excellent": 1.2}
).values

property_type = np.random.choice(
    ["Single Family", "Condo", "Townhouse", "Multi-Family"], N, p=[0.55, 0.20, 0.15, 0.10]
)

# --- Price generation (ground truth formula + noise) ----------------------------
base_price = (
    square_footage * 120
    + bedrooms * 8000
    + bathrooms * 6000
    + lot_size * 2.2
    + garage_spaces * 5000
    + has_pool * 15000
    + has_garden * 4000
    + school_rating * 3500
    - crime_rate * 2000
    - house_age * 400
    + renovated * 12000
)

price = base_price * neighborhood_multiplier * condition_multiplier
price = price / (1 + distance_to_city_center * 0.01)
price = price + np.random.normal(0, 18000, N)  # noise
price = price.clip(50000, None).round(-2)  # round to nearest 100

df = pd.DataFrame({
    "square_footage": square_footage.round(0).astype(int),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "lot_size": lot_size.round(0).astype(int),
    "year_built": year_built,
    "house_age": house_age,
    "neighborhood_quality": neighborhood_quality,
    "distance_to_city_center": distance_to_city_center.round(2),
    "garage_spaces": garage_spaces,
    "has_pool": has_pool,
    "has_garden": has_garden,
    "school_rating": school_rating,
    "crime_rate": crime_rate.round(2),
    "renovated": renovated,
    "condition": condition,
    "property_type": property_type,
    "price": price.astype(int),
})

# Inject a few missing values and a couple of outliers to make cleaning meaningful
missing_idx = np.random.choice(df.index, 40, replace=False)
df.loc[missing_idx[:20], "bathrooms"] = np.nan
df.loc[missing_idx[20:], "school_rating"] = np.nan

outlier_idx = np.random.choice(df.index, 5, replace=False)
df.loc[outlier_idx, "price"] = df.loc[outlier_idx, "price"] * 6

# A few duplicate rows to demonstrate de-duplication in cleaning step
dupes = df.sample(6, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv("data/housing_data.csv", index=False)
print(f"Saved {len(df)} rows to data/housing_data.csv")
