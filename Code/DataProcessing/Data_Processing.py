import pandas as pd
import numpy as np
import os

# Define paths
input_path = os.path.join("Processed Datasets", "Cleaned Dataset.csv")
output_path = os.path.join("Processed Datasets", "Processed Dataset.csv")

if not os.path.exists(input_path):
    input_path = os.path.join("..", "..", "Processed Datasets", "Cleaned Dataset.csv")
    output_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

print(f"Loading data from: {input_path}")
df = pd.read_csv(input_path)

print("\n--- Starting Feature Engineering ---")

# 1. Temporal Feature: is_rush_hour
if 'hour_of_day' in df.columns:
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 16, 17, 18]).astype(int)
    print("Engineered feature: 'is_rush_hour'")

# 2. Spatial Feature: direction_inbound
if 'direction' in df.columns:
    # Assuming 'Inbound' means heading to the city center. Adjust if your raw data uses 0/1 or different text.
    df['direction_inbound'] = df['direction'].apply(lambda x: 1 if 'inbound' in str(x).lower() else 0)
    print("Engineered feature: 'direction_inbound'")

# 3. Environmental Feature: adverse_weather_score
def calculate_adverse_weather(row):
    rain = row.get('rain_mm', 0)
    vis = row.get('visibility', 10000)
    if rain >= 7.6 or vis <= 400:
        return 2 # Severe weather
    elif rain > 0 or vis < 2000:
        return 1 # Mild adverse weather
    return 0 # Clear

if 'rain_mm' in df.columns and 'visibility' in df.columns:
    df['adverse_weather_score'] = df.apply(calculate_adverse_weather, axis=1)
    
    # 4. Interaction Feature: Synergistic effect of rush hour and bad weather
    if 'is_rush_hour' in df.columns:
        df['rush_weather_interaction'] = df['is_rush_hour'] * df['adverse_weather_score']
    print("Engineered features: 'adverse_weather_score' & 'rush_weather_interaction'")

# 5. Cyclical Temporal Features: Trigonometric encoding for days of the week
if 'day_of_week' in df.columns:
    df['dow_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['dow_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    print("Engineered features: 'dow_sin' & 'dow_cos'")

print("\n--- Finalizing Dataset ---")

# 6. STRICT FEATURE SELECTION: Keep only the 6 golden features + target
golden_features = [
    'is_congested', 'is_rush_hour', 'direction_inbound', 
    'adverse_weather_score', 'rush_weather_interaction', 
    'dow_sin', 'dow_cos'
]

# Drop missing values and filter columns
df = df.dropna()
existing_golden = [col for col in golden_features if col in df.columns]
df_final = df[existing_golden]

# Save the fully processed dataset
df_final.to_csv(output_path, index=False)
print(f"Successfully saved {len(df_final)} rows with exactly {len(df_final.columns)} columns.")
print(f"Columns kept: {list(df_final.columns)}")
print(f"Saved to: {output_path}")