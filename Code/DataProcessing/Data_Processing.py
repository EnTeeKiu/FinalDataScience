import pandas as pd
import os

# Define paths
# Adjust paths depending on where the script is run from
input_path = os.path.join("Processed Datasets", "Cleaned Dataset.csv")
output_path = os.path.join("Processed Datasets", "Processed Dataset.csv")

# Fallback if run from inside Code/DataProcessing
if not os.path.exists(input_path):
    input_path = os.path.join("..", "..", "Processed Datasets", "Cleaned Dataset.csv")
    output_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

print(f"Loading data from: {input_path}")
df = pd.read_csv(input_path)

# 1. Drop unnecessary rows (e.g., rows with missing values)
initial_len = len(df)
df = df.dropna()
print(f"Dropped {initial_len - len(df)} rows with missing values.")

# 2. Drop columns to avoid data leakage
columns_to_drop = [
    'speed_deficit',
    'magnitude', 
    'incident_type', 
    'travel_time_s', 
    'free_flow_time_s', 
    'speed_limit_baseline', 
    'current_speed', 
    'frc_class', 
    'is_weekend', 
    'speed_ratio',
    'speed_ratio_proxy' # Alternative name for speed ratio
]

# Only drop columns that actually exist in the dataframe to avoid errors
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Dropped columns: {existing_cols_to_drop}")

# Save the processed dataset
df.to_csv(output_path, index=False)
print(f"Successfully saved processed dataset to: {output_path}")
