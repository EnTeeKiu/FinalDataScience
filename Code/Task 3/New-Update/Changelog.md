# Unsupervised Learning v02 Changelog

This document outlines the key differences and improvements between the original unsupervised learning pipeline and the updated `Unsupervised_Learning_v02.ipynb` notebook.

## 1. Feature Scaling: `StandardScaler` ➔ `MinMaxScaler`
- **Original:** Used `StandardScaler` (mean=0, variance=1) which can distort variables that have a strict ordinal or binary structure.
- **v02 Update:** Changed to `MinMaxScaler` to strictly bound features to the `[0, 1]` interval. This preserves the binary nature of `is_rush_hour` and the ordinal severity of `adverse_weather_score`, resulting in cleaner Euclidean distance boundaries for K-Means.

## 2. Optimal K Selection: `K=6` ➔ `K=4`
- **Original:** Evaluated `K=6` as the optimal number of clusters using `n_init=10`.
- **v02 Update:** Comprehensive Silhouette Score and WCSS (Elbow) analysis on the full 6-dimensional dataset revealed a prominent peak and elbow at **$K=4$** (Silhouette score > 0.41). K-Means stability was also improved by increasing random initializations to `n_init=20`.

## 3. Dimensionality Reduction & Visualization
- **Original:** Used only **PCA** (Principal Component Analysis) for 2D visual mapping.
- **v02 Update:** Retained PCA but added **t-SNE** (t-Distributed Stochastic Neighbor Embedding). While PCA only captures linear variance, t-SNE perfectly unfurls the non-linear, daily cyclical structures of the traffic data in 2D space.

## 4. Cluster Interpretation & Profiling
- **Original:** Profiled 6 clusters with overlapping temporal contexts.
- **v02 Update:** Reduced to 4 foundational, highly distinct operational states of the bridge:
  - **Cluster 0:** Off-peak, inbound (Non-rush inbound traffic)
  - **Cluster 1:** Off-peak, outbound (Non-rush outbound traffic)
  - **Cluster 2:** Rush hour, clear weather (Heavy rush hour traffic)
  - **Cluster 3:** Rush hour, adverse weather (Heavy rush hour compounded by bad weather)
