import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

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

print("\n--- Starting Data Processing ---")

# 1. Drop unnecessary rows (e.g., rows with missing values)
initial_len = len(df)
df = df.dropna()
print(f"Dropped {initial_len - len(df)} rows with missing values.")

# 2. Drop columns to avoid data leakage (Combined)
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
    'speed_ratio_proxy',
    'route_name', 
    'timestamp'
    # Ensuring route_delay_s and hour_of_day are explicitly NOT dropped
]

# Only drop columns that actually exist in the dataframe to avoid errors
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Dropped columns: {existing_cols_to_drop}")

print("\n--- Starting Feature Engineering ---")

# 1. Create is_rush_hour
if 'hour_of_day' in df.columns:
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 16, 17]).astype(int)
    print("Engineered feature: 'is_rush_hour'")

# 2. Weather Severity Parsing
def parse_weather_severity(weather_string):
    if pd.isna(weather_string):
        return "Clear"
    w_str = str(weather_string)
    if 'Rain' in w_str:
        return 'Rain'
    elif 'Overcast' in w_str:
        return 'Overcast'
    elif 'Partially cloudy' in w_str:
        return 'Partially cloudy'
    else:
        return 'Clear'

if 'weather' in df.columns:
    df['weather'] = df['weather'].apply(parse_weather_severity)
    print("Simplified 'weather' column.")

# 3. Adverse Weather Score
def calculate_adverse_weather_frc2(row):
    rain = row.get('rain_mm', 0)
    vis = row.get('visibility', 10000)
    
    if rain >= 7.6 or vis <= 400:
        return 2
    elif rain > 0 or vis < 2000:
        return 1
    else:
        return 0

if 'rain_mm' in df.columns and 'visibility' in df.columns:
    df['adverse_weather_score'] = df.apply(calculate_adverse_weather_frc2, axis=1)
    if 'is_rush_hour' in df.columns:
        df['rush_weather_interaction'] = df['is_rush_hour'] * df['adverse_weather_score']
    print("Engineered feature: 'adverse_weather_score' & 'rush_weather_interaction'")

if 'incident_type' in df.columns:
    df['incident_type'] = df['incident_type'].fillna('None')

# 4. Encode categorical columns
le = LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f"Encoded categorical columns: {list(object_cols)}")

# Save the fully processed dataset
df.to_csv(output_path, index=False)
print(f"\nSuccessfully saved fully processed and engineered dataset to: {output_path}")
