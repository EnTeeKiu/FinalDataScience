import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File names
raw_data_file = "Processed Dataset.csv"
preprocessed_data_file = "Preprocessed_Data_VinhTuy.csv"

# Define file paths
raw_data_path = os.path.join("Processed Datasets", raw_data_file)
output_dir = "Data Visualization"

# Fallback paths
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join("..", "..", "Processed Datasets", raw_data_file)
    output_dir = os.path.join("..", "..", "Data Visualization")

if not os.path.exists(raw_data_path):
    local_abs_path = r"..." # Input local storage destination
    raw_data_path = local_abs_path if os.path.exists(local_abs_path) else raw_data_file

os.makedirs(output_dir, exist_ok=True)
sns.set_theme(style="whitegrid")

# ==========================================
# Phase 1: Raw Data Visualizations
# ==========================================
print(f"Loading raw data for exploratory plots: {raw_data_path}")
df_raw = pd.read_csv(raw_data_path)

# Plot 1: Target Variable (is_congested)
if 'is_congested' in df_raw.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_congested', data=df_raw, palette='Set2')
    plt.title('Distribution of Target Variable (is_congested)')
    plt.xlabel('Is Congested (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, '1_target_distribution.png'), bbox_inches='tight')
    plt.close()

# Plot 2: Route Delay Distribution
if 'route_delay_s' in df_raw.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_raw['route_delay_s'], bins=50, kde=True, color='#87CEEB', alpha=0.6, edgecolor='black', linewidth=0.7)
    plt.title('Distribution of Route Delay (seconds)')
    plt.xlabel('Route Delay (s)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, '2_route_delay_distribution.png'), bbox_inches='tight')
    plt.close()

# Plot 3: Route Delay by Hour
if 'hour_of_day' in df_raw.columns and 'route_delay_s' in df_raw.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hour_of_day', y='route_delay_s', data=df_raw, palette='muted')
    plt.title('Route Delay by Hour of the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Route Delay (s)')
    plt.savefig(os.path.join(output_dir, '3_delay_by_hour.png'), bbox_inches='tight')
    plt.close()

# Plot 4: Temperature vs Humidity colored by Congestion
if all(col in df_raw.columns for col in ['temp', 'humidity', 'is_congested']):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temp', y='humidity', hue='is_congested', data=df_raw, palette={0: '#1f77b4', 1: '#d62728'}, alpha=0.8, s=40)
    plt.title('Temperature vs Humidity (Colored by Congestion)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.savefig(os.path.join(output_dir, '4_temp_vs_humidity_scatter.png'), bbox_inches='tight')
    plt.close()

print("✅ Exploratory visualizations generated.")

# ==========================================
# Phase 2: Correlation Heatmap
# ==========================================
print("\nGenerating Correlation Heatmap...")
if len(df_raw.columns) > 1:
    plt.figure(figsize=(14, 12))
    
    # Calculate correlation for numerical columns only
    numerical_cols = df_raw.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    corr_matrix = df_raw[numerical_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".0%",
        cmap='coolwarm',
        cbar=True,
        square=True,
        annot_kws={"size": 9}
    )

    plt.title('Correlation Heatmap of Features (Percentage Format)', fontsize=16, pad=20)
    plt.savefig(os.path.join(output_dir, '5_correlation_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("✅ Correlation Heatmap successfully generated.")
else:
    print("⚠️ Error: Not enough columns left to construct a heatmap.")

print("\n🎉 Visualization Complete!")
