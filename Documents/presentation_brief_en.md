# Presentation and Report Brief: Hanoi Traffic Congestion Analysis (Vinh Tuy Bridge)

Here is a humanized, structured, and academically polished brief written in English. You can copy and paste these sections directly into your slides and report.

---

## 1. Project Overview

### Why / Purpose?
*   **The Problem:** Urban traffic congestion is a massive daily headache in Hanoi, with critical bottlenecks like the **Vinh Tuy Bridge** causing severe delays, economic loss, and environmental stress.
*   **Our Goal:** We wanted to move away from guesswork. The purpose of this project is to analyze, predict, and segment traffic states using real-world spatial, temporal, and environmental data.
*   **Real-world Value:** By identifying the exact combinations of conditions (like time of day and severe weather) that trigger gridlock, we can help city planners optimize traffic flow and help commuters make smarter travel decisions.

### Action?
*   **End-to-End Pipeline:** We built a complete data science pipeline:
    1.  **Collaborative Data Scraping:** Collected 1 year of historical traffic and weather data.
    2.  **Rigorous Data Cleaning & Feature Engineering:** Structured raw inputs and engineered key physical features.
    3.  **Supervised Machine Learning:** Applied and compared **Logistic Regression** and **Support Vector Machine (SVM)** to predict congestion.
    4.  **Unsupervised Clustering:** Used **K-Means Clustering** and **PCA visualization** to discover and profile hidden traffic contexts.

---

## 2. Data Processing

### Data Collection Method
*   **Collaborative Scraping:** Our team split the work to crawl a 1-year historical dataset.
*   **Traffic Logs:** Scraped travel times, traffic delays, and free-flow speed baseline across the Vinh Tuy Bridge deck span using the **TomTom Routing API**.
*   **Weather Logs:** Scraped hourly weather data (precipitation, humidity, temperature, and visibility) aligned with our traffic timestamps using the **Visual Crossing Weather API**.

### Data Pre-processing
*   **Temporal Alignment:** Implemented a nearest-hour matching algorithm to perfectly align hourly weather records with traffic log timestamps.
*   **Spatial Directional Mapping:** Extracted and mapped route endpoints to distinguish between **Inbound** (heading to the city center) and **Outbound** traffic.

### Data Cleaning (Clean Dataset)
*   **Handling Missing Values:** Dropped rows missing the target variable (`is_congested`). For missing numerical features, we used a highly precise **Grouped Median Imputation** (grouping by direction, day of week, and hour of day) to preserve natural data distributions without introducing global bias.
*   **Deduping:** Removed duplicate logs to ensure statistical independence.
*   **Outcome:** Retained a robust, balanced dataset of **7,320 valid samples**, split equally (50% Train, 50% Test) for model evaluation.

### Data Visualization
*   **Class Imbalance Chart:** Plotted the distribution of `is_congested` to show the jury that congestion is relatively rare, justifying the use of **F1-Score** instead of just raw Accuracy.
*   **Weather Impact Plot:** Bar chart proving a direct relationship between our custom weather severity score and the probability of traffic jams.
*   **Interaction Effect Plot:** Multi-hue bar chart demonstrating how bad weather acts as a "disaster multiplier" when combined with rush hour.
*   **Correlation Heatmap:** A Pearson correlation matrix of engineered features to confirm we minimized multicollinearity before training.

---

## 3. Feature Engineering

To help our models capture real-world traffic logic, we engineered 6 "golden features":
1.  **`is_rush_hour` (Binary: 0, 1):** Captures peak commuting hours (7-8 AM, 4-6 PM).
2.  **`direction_inbound` (Binary: 0, 1):** Represents spatial flow (1 = entering city center, 0 = exiting).
3.  **`adverse_weather_score` (Ordinal: 0, 1, 2):** Quantifies weather severity (0 = Clear, 1 = Rain/Fog, 2 = Severe rain/Fog).
4.  **`rush_weather_interaction` (Continuous):** An interaction term (`is_rush_hour` * `adverse_weather_score`) capturing the compounded effect of storm-ridden rush hours.
5.  **`dow_sin` & `dow_cos` (Continuous, -1 to 1):** Trigonometric sine and cosine encodings of the day of the week to capture the cyclical nature of weekly travel behavior.

---

## 4. Supervised Models

*To keep our evaluations fair and prevent **Data Leakage**, we split the dataset into Train/Test first, and fit our `StandardScaler` strictly on the training set.*

### Logistic Regression
*   **How it works:** Classifies traffic status using a sigmoid activation function to map linear combinations of features to a probability between 0 and 1.
*   **Results:**
    *   **Testing Accuracy:** 80.25%
    *   **Testing F1-Score:** 54.56%
    *   **Testing Precision:** 59.29%
    *   **Testing Recall:** 50.52%

### Support Vector Machine (SVM)
*   **How it works:** Finds the optimal hyperplane in a standardized 6-dimensional space that maximizes the margin between congested and non-congested classes.
*   **Results:**
    *   **Testing Accuracy:** 80.96%
    *   **Testing F1-Score:** 60.82%
    *   **Testing Precision:** 58.80%
    *   **Testing Recall:** 62.98%

---

## 5. Unsupervised Model (K-Means Clustering)

*   **Approach:** We applied **K-Means clustering** directly on our 6 standardized features (excluding the target `is_congested`) to let the algorithm find natural traffic states.
*   **Optimal K Selection:** Determined $K=6$ as the optimal number of clusters using a combination of the **Elbow Method** (WCSS) and **Silhouette Analysis**.
*   **Visualization & Dimensionality Reduction:**
    *   To view our 6D clusters in a 2D plane, we projected the data using **Principal Component Analysis (PCA)**.
    *   *The PCA Overlap:* On the 2D PCA plot, the clusters display some overlap. This is a normal mathematical artifact because the first 2 PCs capture **45.46%** of the total variance, meaning 54.54% of the feature variance is omitted in 2D. However, the clusters are mathematically distinct in the original 6D space (as proven by our cluster profiling).

---

## 6. Evaluation and Key Takeaways

### Supervised Insights
*   **Winner:** **SVM outperformed Logistic Regression** (F1-Score: 60.82% vs 54.56%).
*   **Why?** SVM achieved a much higher Recall (62.98% vs 50.52%). In traffic management, missing a real traffic jam (False Negative) is worse than predicting a minor delay (False Positive). SVM is much better at capturing these critical congestion events.

### Unsupervised Insights (Cluster Profiling)
Our K-Means model successfully isolated 6 distinct, real-world traffic contexts:
*   **Cluster 1 (Morning Inbound Commute):** Peak rush hour, strictly heading towards the city center (`direction_inbound = 1.0`), clear weather. Congestion rate is the highest at **57.32%**.
*   **Cluster 5 (Evening Outbound Commute):** Peak rush hour, strictly exiting the city (`direction_inbound = 0.0`), clear weather. Congestion rate is **49.57%**.
*   **Cluster 2 (Stormy Peak Hours):** Rush hour traffic compounded by bad weather (`adverse_weather_score = 1.04`, interaction = 1.04). Congestion rate is high (**52.67%**) across both directions.
*   **Cluster 4 (Off-Peak Bad Weather):** Adverse weather during quiet hours. Congestion remains very low (**4.63%**), proving that weather alone doesn't cause gridlock without peak volume.
*   **Cluster 0 & 3 (Off-Peak Clear Days):** Smooth flowing traffic on weekdays and weekends respectively, with **0% to 7%** congestion probability.

### Methodology Takeaway (For the Jury)
*   Standardizing data *before* train/test splitting leads to **data leakage**, giving overly optimistic but fake metrics.
*   Clustering on PCA-reduced data (when variance explained is low) ruins spatial features. Clustering directly on the 6D space is what allowed us to separate inbound vs outbound rush hour profiles.

---

## 7. References

*   **TomTom Routing API Developer Portal:** Historical traffic flow and speed calculations.
*   **Visual Crossing Weather API:** Historical weather records and meteorological variables.
*   **Scikit-Learn Library:** Implementations of `StandardScaler`, `LogisticRegression`, `LinearSVC`, `KMeans`, and `PCA`.
*   **Pandas & NumPy:** Data cleaning, interpolation, and matrix operations.
