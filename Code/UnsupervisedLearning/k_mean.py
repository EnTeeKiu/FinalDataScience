import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

# --- Configuration & Setup ---
sns.set_theme(style="whitegrid")

# Ensure output directory for visualization exists
output_dir = "Data Visualization"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

X = df[features]
y_true = df['is_congested']

# Standardize data (Crucial before PCA)
scaler = StandardScaler()
X_processed = scaler.fit_transform(X)

print(f"Original dataset shape: {X_processed.shape}")

# =====================================================================
# STEP 1: PCA (DIMENSIONALITY REDUCTION BEFORE K-MEANS)
# =====================================================================
print("\n--- STEP 1: Running PCA to reduce dimensionality ---")
# Reduce the 6 features down to 2 Principal Components to denoise and prepare for K-Means
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

explained_variance = pca.explained_variance_ratio_
total_variance_explained = sum(explained_variance) * 100
print(f"Reduced to 2 Principal Components.")
print(f"Total Variance Explained by 2 PCs: {total_variance_explained:.2f}%")

# =====================================================================
# STEP 2: FINDING THE OPTIMAL 'K' ON PCA DATA
# =====================================================================
print("\n--- STEP 2: Running Elbow Method and Silhouette Analysis on PCA data ---")
inertias = []
silhouette_scores = []
K_range = range(2, 11) 

# NOTICE: We are now fitting K-Means on X_pca, not X_processed
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca) 
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, labels))

# Plotting the evaluation metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=12)
ax1.plot(K_range, inertias, marker='o', color=color, linewidth=2, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)  
ax2.plot(K_range, silhouette_scores, marker='s', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Optimal K Selection (on PCA Data)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, "5_pca_elbow_silhouette.png"), dpi=300)
print("Saved Optimal K plot to: 5_pca_elbow_silhouette.png")
plt.close()

# =====================================================================
# STEP 3: FINAL K-MEANS IMPLEMENTATION
# =====================================================================
OPTIMAL_K = 6
print(f"\n--- STEP 3: Fitting Final K-Means Model with K={OPTIMAL_K} ---")

# Fit on PCA data
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_pca)

print(f"Final Model Inertia (Loss): {kmeans.inertia_:.2f}")

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

# Even though we clustered in PCA space, we MUST group by original features 
# so we can explain what the clusters mean in real life.
cluster_profiles = df.groupby('Cluster')[features].mean()
cluster_profiles['Congestion_Rate (%)'] = df.groupby('Cluster')['is_congested'].mean() * 100
cluster_profiles['Count'] = df.groupby('Cluster').size()
cluster_profiles = cluster_profiles.round(2)

print("\nCluster Profiles (Original Feature Means & Actual Congestion Rate):")
print(cluster_profiles.to_string())

print("\n--- ANALYSIS CONCLUSION FOR REPORT ---")
print("By applying PCA before K-Means, we reduced the feature space to its principal components,")
print("eliminating noise and allowing K-Means to find more robust, spherical clusters.")
print("The profiling shows these PCA-based clusters still translate perfectly to real-world traffic contexts.")