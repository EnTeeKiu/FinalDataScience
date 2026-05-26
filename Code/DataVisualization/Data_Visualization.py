import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Setup ---
# Use the final Processed Dataset that contains our golden features
input_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
if not os.path.exists(input_path):
    input_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

output_dir = os.path.join("Data Visualization", "Task 1")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"Loading data for visualization from: {input_path}")
df = pd.read_csv(input_path)

# Ensure styling is clean and professional for the report
sns.set_theme(style="whitegrid")

# =====================================================================
# PLOT 1: Class Imbalance (Target Distribution)
# Purpose: Show the jury why we need F1-Score instead of just Accuracy
# =====================================================================
plt.figure(figsize=(8, 6))
custom_colors = ['#4A90E2', '#D9534F'] # Soft Blue (Normal), Soft Red (Congested)
ax = sns.countplot(x='is_congested', data=df, palette=custom_colors)
plt.title('Distribution of Traffic Congestion States', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Traffic Flow State', fontsize=12, labelpad=10)
plt.ylabel('Number of Records', fontsize=12, labelpad=10)

# Set custom x-tick labels directly to make it clean
ax.set_xticks([0, 1])
ax.set_xticklabels(['Normal Flow', 'Congested'])

# Add descriptive legend manually to avoid duplicate bar errors
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#4A90E2', label='0: Normal Flow')
red_patch = mpatches.Patch(color='#D9534F', label='1: Congested Traffic')
ax.legend(handles=[blue_patch, red_patch], title='Traffic Status', loc='upper right')

# Add value labels on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "1_target_distribution.png"), dpi=300)
print("Generated Plot 1: Target Distribution")
plt.close()

# =====================================================================
# PLOT 2: Congestion Rate by Adverse Weather Score
# Purpose: Prove that our custom Heuristic Score actually works
# =====================================================================
if 'adverse_weather_score' in df.columns:
    plt.figure(figsize=(8, 6))
    weather_colors = ['#8FBC8F', '#F4A460', '#D9534F'] # Sage Green (Clear), Sandy Orange (Mild), Soft Red (Severe)
    ax = sns.barplot(x='adverse_weather_score', y='is_congested', data=df, palette=weather_colors, errorbar=None)
    plt.title('Congestion Probability by Weather Severity', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Weather Severity Score', fontsize=12, labelpad=10)
    plt.ylabel('Probability of Congestion (%)', fontsize=12, labelpad=10)
    plt.ylim(0, 1) # Set Y-axis from 0 to 1 (0% to 100%)
    
    # Custom x-tick labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Clear', 'Mild', 'Severe'])
    
    # Add descriptive legend
    green_patch = mpatches.Patch(color='#8FBC8F', label='0: Clear / Normal')
    orange_patch = mpatches.Patch(color='#F4A460', label='1: Light Rain / Low Visibility')
    red_patch = mpatches.Patch(color='#D9534F', label='2: Heavy Rain / Dense Fog')
    ax.legend(handles=[green_patch, orange_patch, red_patch], title='Weather Severity', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_weather_impact.png"), dpi=300)
    print("Generated Plot 2: Weather Impact on Congestion")
    plt.close()

# =====================================================================
# PLOT 3: The Interaction Effect (Rush Hour + Weather)
# Purpose: Visually demonstrate the "Synergistic Effect" for Task 2
# =====================================================================
if 'is_rush_hour' in df.columns and 'adverse_weather_score' in df.columns:
    plt.figure(figsize=(10, 6))
    weather_colors = ['#8FBC8F', '#F4A460', '#D9534F']
    ax = sns.barplot(x='is_rush_hour', y='is_congested', hue='adverse_weather_score', data=df, palette=weather_colors, errorbar=None)
    plt.title('Interaction Effect: Rush Hour vs. Weather Severity', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Rush Hour Period', fontsize=12, labelpad=10)
    plt.ylabel('Probability of Congestion (%)', fontsize=12, labelpad=10)
    plt.ylim(0, 1)
    
    # Custom x-tick labels to make it very clean
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Off-Peak Hour', 'Rush Hour'])
    
    # Add descriptive legend
    green_patch = mpatches.Patch(color='#8FBC8F', label='0: Clear / Normal')
    orange_patch = mpatches.Patch(color='#F4A460', label='1: Light Rain / Low Visibility')
    red_patch = mpatches.Patch(color='#D9534F', label='2: Heavy Rain / Dense Fog')
    ax.legend(handles=[green_patch, orange_patch, red_patch], title='Weather Severity', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_interaction_effect.png"), dpi=300)
    print("Generated Plot 3: Interaction Effect")
    plt.close()

# =====================================================================
# PLOT 4: Correlation Heatmap of Golden Features
# Purpose: Prove to the jury that Multicollinearity has been eliminated
# =====================================================================
plt.figure(figsize=(10, 8))
# Select only the features we actually feed into the models
features_to_plot = [
    'is_congested', 'is_rush_hour', 'direction_inbound', 
    'adverse_weather_score', 'rush_weather_interaction', 
    'dow_sin', 'dow_cos'
]
# Filter out any features that might be missing just in case
existing_features = [col for col in features_to_plot if col in df.columns]

corr_matrix = df[existing_features].corr()

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True,
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Pearson Correlation Matrix (Golden Features)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4_correlation_heatmap.png"), dpi=300)
print("Generated Plot 4: Correlation Heatmap")
plt.close()

print("\n--- Visualization Pipeline Completed Successfully ---")
print(f"Check the '{output_dir}' folder for the new PNG files.")