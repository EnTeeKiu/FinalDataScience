import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
input_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
output_dir = "Data Visualization"

# Fallback if run from inside Code/DataVisualization
if not os.path.exists(input_path):
    input_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")
    output_dir = os.path.join("..", "..", "Data Visualization")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Loading data from: {input_path}")
df = pd.read_csv(input_path)

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Feature Distribution: Target Variable (is_congested)
plt.figure(figsize=(8, 6))
sns.countplot(x='is_congested', data=df, palette='Set2')
plt.title('Distribution of Target Variable (is_congested)')
plt.xlabel('Is Congested (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, '1_target_distribution.png'), bbox_inches='tight')
plt.close()

# 2. Feature Distribution: Route Delay (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['route_delay_s'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Route Delay (seconds)')
plt.xlabel('Route Delay (s)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, '2_route_delay_distribution.png'), bbox_inches='tight')
plt.close()

# 3. Relationships: Route Delay by Hour of Day (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour_of_day', y='route_delay_s', data=df, palette='muted')
plt.title('Route Delay by Hour of the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Route Delay (s)')
plt.savefig(os.path.join(output_dir, '3_delay_by_hour.png'), bbox_inches='tight')
plt.close()

# 4. Relationships: Scatter Plot of Temperature vs Humidity colored by Congestion
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='humidity', hue='is_congested', data=df, palette='coolwarm', alpha=0.6)
plt.title('Temperature vs Humidity (Colored by Congestion)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.savefig(os.path.join(output_dir, '4_temp_vs_humidity_scatter.png'), bbox_inches='tight')
plt.close()

# 5. Correlations: Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numerical_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig(os.path.join(output_dir, '5_correlation_heatmap.png'), bbox_inches='tight')
plt.close()

print(f"Visualizations successfully generated and saved in the '{output_dir}' folder.")
