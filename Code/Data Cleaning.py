import os
import pandas as pd
import numpy as np

# Config file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Raw Datasets", "VINH_TUY.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed Datasets")
TARGET_COL = "is_congested"

def clean_vinh_tuy_data():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    initial_rows = len(df)
    
    # 1. Drop rows without the target variable
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    
    import holidays
    vn_holidays = holidays.VN()
    
    # 2. Derive basic features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: d in vn_holidays).astype(int)
    
    if "speed_limit_baseline" in df.columns and "current_speed" in df.columns:
        df["speed_deficit"] = df["speed_limit_baseline"] - df["current_speed"]
    
    # 3. Time-based Interpolation for numeric values
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Sort data chronologically so interpolation makes sense
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp")
        
    # Interpolate linearly, grouped by direction if available (to prevent blending inbound/outbound)
    if "direction" in df.columns:
        df[num_cols] = df.groupby("direction")[num_cols].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    else:
        df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
        
    # Fallback to median just in case an entire column was NaN
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # 4. Fill missing categorical values with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            
    # 5. Remove duplicates
    df = df.drop_duplicates()
    
    print(f"Cleaning: {initial_rows} -> {len(df)} rows after cleaning.")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, "Cleaned Dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")

if __name__ == "__main__":
    clean_vinh_tuy_data()
