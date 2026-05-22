import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Setup ---
# Use the final Processed Dataset that contains our golden features
input_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
if not os.path.exists(input_path):
    input_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

output_dir = "Data Visualization"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Loading data for visualization from: {input_path}")
df = pd.read_csv(input_path)

# Ensure styling is clean and professional for the report
sns.set_theme(style="whitegrid")

# =====================================================================
# PLOT 1: Class Imbalance (Target Distribution)
# Purpose: Show the jury why we need F1-Score instead of just Accuracy
# =====================================================================
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='is_congested', data=df, palette='Set2')
plt.title('Distribution of Traffic Congestion (Class Imbalance)', fontsize=14, fontweight='bold')
plt.xlabel('Is Congested (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Number of Records', fontsize=12)

# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
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
    sns.barplot(x='adverse_weather_score', y='is_congested', data=df, palette='Reds', errorbar=None)
    plt.title('Congestion Probability by Weather Severity', fontsize=14, fontweight='bold')
    plt.xlabel('Adverse Weather Score (0=Clear, 1=Mild, 2=Severe)', fontsize=12)
    plt.ylabel('Probability of Congestion (%)', fontsize=12)
    plt.ylim(0, 1) # Set Y-axis from 0 to 1 (0% to 100%)
    
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
    sns.barplot(x='is_rush_hour', y='is_congested', hue='adverse_weather_score', data=df, palette='YlOrRd', errorbar=None)
    plt.title('Interaction Effect: Rush Hour x Weather Severity', fontsize=14, fontweight='bold')
    plt.xlabel('Is Rush Hour? (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Probability of Congestion (%)', fontsize=12)
    plt.legend(title='Weather Score')
    plt.ylim(0, 1)
    
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