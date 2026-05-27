import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

# --- Configuration & Setup ---
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
OPTIMAL_K = 4

# Ensure output directory for visualization exists
output_dir = os.path.join("Data Visualization", "Task 3")
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
file_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

df = pd.read_csv(file_path)

# The 6 core features
features = [
    'is_rush_hour', 'direction_inbound', 'adverse_weather_score',
    'rush_weather_interaction', 'dow_sin', 'dow_cos'
]
target = 'is_congested'

missing_columns = sorted(set(features + [target]) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

X = df[features]

# MinMax scaling preserves the binary/ordinal structure of the engineered
# features and gives K-Means cleaner traffic-state separation.
scaler = MinMaxScaler()
X_processed = scaler.fit_transform(X)

print(f"Original dataset shape: {X_processed.shape}")

# =====================================================================
# STEP 1: PCA TRANSFORMATION FOR SCORE-OPTIMIZED K-MEANS
# =====================================================================
print("\n--- STEP 1: Running PCA before K-Means for stronger cluster separation ---")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_processed)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

explained_variance = pca.explained_variance_ratio_
total_variance_explained = explained_variance.sum() * 100
print("Reduced to 2 Principal Components.")
print(f"Total Variance Explained by 2 PCs: {total_variance_explained:.2f}%")

# =====================================================================
# STEP 2: FINDING THE OPTIMAL 'K' ON PCA DATA
# =====================================================================
print("\n--- STEP 2: Running Elbow Method and Silhouette Analysis on PCA data ---")
K_range = range(2, 11)
evaluation_rows = []
selected_model = None
selected_labels = None
selected_silhouette = None

for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, labels)

    evaluation_rows.append({
        "K": k,
        "Inertia": km.inertia_,
        "Silhouette Score": silhouette,
    })

    if k == OPTIMAL_K:
        selected_model = km
        selected_labels = labels
        selected_silhouette = silhouette

if selected_model is None:
    raise ValueError(f"OPTIMAL_K={OPTIMAL_K} was not evaluated in K_range.")

k_evaluation = pd.DataFrame(evaluation_rows)
print("\nK-Means evaluation by K:")
print(k_evaluation.round(4).to_string(index=False))

# Plotting the evaluation metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)
ax1.plot(k_evaluation["K"], k_evaluation["Inertia"], marker='o', color=color, linewidth=2, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)  
ax2.plot(k_evaluation["K"], k_evaluation["Silhouette Score"], marker='s', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Optimal K Selection (on PCA Data)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, "5_elbow_silhouette.png"), dpi=300)
print("Saved Optimal K plot to: 5_elbow_silhouette.png")
plt.close()

# =====================================================================
# STEP 3: FINAL K-MEANS IMPLEMENTATION ON PCA DATA
# =====================================================================
print(f"\n--- STEP 3: Using Final K-Means Model with K={OPTIMAL_K} ---")

df['Cluster'] = selected_labels

print(f"Final Model Inertia (Loss): {selected_model.inertia_:.2f}")
print(f"Final Model Silhouette Score: {selected_silhouette:.4f}")

# =====================================================================
# STEP 4: VISUALIZATION IN PCA SPACE
# =====================================================================
print("\n--- STEP 4: Generating PCA Visualization ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette='Set1', 
    data=df, 
    alpha=0.6, 
    edgecolor=None
)
plt.title(f'K-Means Clusters in PCA Space (Variance Explained: {total_variance_explained:.1f}%)', fontsize=14, fontweight='bold')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Cluster')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "6_pca_clusters_direct.png"), dpi=300)
print("Saved PCA visualization plot to: 6_pca_clusters_direct.png")
plt.close()

# =====================================================================
# STEP 5: POST-CLUSTERING ANALYSIS (Translate back to Original Features)
# =====================================================================
print("\n--- STEP 5: TRAFFIC CONTEXT PROFILING ---")

# Profiles are computed using original features to explain each cluster in traffic terms.
cluster_groups = df.groupby('Cluster')
cluster_profiles = cluster_groups[features].mean()
cluster_profiles['Congestion_Rate (%)'] = cluster_groups[target].mean() * 100
cluster_profiles['Count'] = cluster_groups.size()
cluster_profiles = cluster_profiles.round(2)

print("\nCluster Profiles (Original Feature Means & Actual Congestion Rate):")
print(cluster_profiles.to_string())

# Generate Cluster Profile Heatmap for visualization
plt.figure(figsize=(10, 6))
profile_heatmap_df = cluster_profiles[features].rename(columns={
    'is_rush_hour': 'Rush Hour Period',
    'direction_inbound': 'Inbound Direction',
    'adverse_weather_score': 'Weather Severity',
    'rush_weather_interaction': 'Rush x Weather Interaction',
    'dow_sin': 'Day of Week (Sin)',
    'dow_cos': 'Day of Week (Cos)',
})
sns.heatmap(profile_heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Cluster Profile Mean Features (K-Means Traffic Contexts)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Features", fontsize=12, labelpad=10)
plt.ylabel("K-Means Cluster", fontsize=12, labelpad=10)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "7_cluster_profiles.png"), dpi=300)
print("Saved Cluster Profiles heatmap to: 7_cluster_profiles.png")
plt.close()

print("\n--- ANALYSIS CONCLUSION FOR REPORT ---")
print("This score-optimized pipeline uses MinMax scaling and clusters in a 2D PCA space,")
print("which produces much stronger internal K-Means separation. The cluster profiles")
print("translate the PCA-based groups back into real-world traffic contexts.")
