# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# 2. LOAD DATA
df = pd.read_csv("data/nyc_taxi_final/nyc_taxi_trip_duration.csv")

# 3. FEATURE ENGINEERING

# Convert datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.weekday
df["month"] = df["pickup_datetime"].dt.month


# Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


df["distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Encode categorical
df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})

# Drop unused columns
df = df.drop(columns=["id", "dropoff_datetime", "pickup_datetime"])

# 4. SPLIT FEATURES/TARGET
X = df.drop(columns=["trip_duration"])
y = df["trip_duration"]

# 5. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. PIPELINE
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,  # L1
        reg_lambda=1.0,  # L2
        random_state=42
    ))
])

# 7. TRAIN MODEL
pipeline.fit(X_train, y_train)

# 8. EVALUATION
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# SHOW SAMPLE PREDICTIONS
# Create comparison dataframe
results_df = X_test.copy()
results_df["Actual_Trip_Duration"] = y_test.values
results_df["Predicted_Trip_Duration"] = y_pred

# Show few rows
print("\n===== SAMPLE PREDICTIONS =====")
print(results_df.head(10))