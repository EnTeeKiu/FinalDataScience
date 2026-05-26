import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# --- Load and Split Data ---
file_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

df = pd.read_csv(file_path)

features = [
    'is_rush_hour', 'direction_inbound', 'adverse_weather_score',
    'rush_weather_interaction', 'dow_sin', 'dow_cos'
]
target = 'is_congested'

X = df[features]
y = df[target].astype(int)

# Split 50% Train - 50% Test (stratify=y for class balance)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# --- Train Models ---
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_test_pred_log = log_reg.predict(X_test)
# Decision function/probabilities for ROC
y_score_log = log_reg.predict_proba(X_test)[:, 1]

# Linear SVM (using LinearSVC)
svm = LinearSVC(max_iter=10000, random_state=42)
svm.fit(X_train, y_train)
y_test_pred_svm = svm.predict(X_test)
# Decision function/margins for ROC
y_score_svm = svm.decision_function(X_test)

# Ensure output directory exists
output_dir = os.path.join("Data Visualization", "Task 2")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

# =====================================================================
# CHART 1: Confusion Matrices Heatmap
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_log = confusion_matrix(y_test, y_test_pred_log)
cm_svm = confusion_matrix(y_test, y_test_pred_svm)

# Plot Logistic Regression Confusion Matrix
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title("Logistic Regression Confusion Matrix", fontsize=14, fontweight="bold", pad=10)
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)
axes[0].set_xticklabels(["Normal Flow", "Congested"])
axes[0].set_yticklabels(["Normal Flow", "Congested"])

# Plot SVM Confusion Matrix
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges", ax=axes[1], cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title("Support Vector Machine Confusion Matrix", fontsize=14, fontweight="bold", pad=10)
axes[1].set_xlabel("Predicted Label", fontsize=12)
axes[1].set_ylabel("True Label", fontsize=12)
axes[1].set_xticklabels(["Normal Flow", "Congested"])
axes[1].set_yticklabels(["Normal Flow", "Congested"])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "8_confusion_matrices.png"), dpi=300)
print("Saved Confusion Matrices plot to: 8_confusion_matrices.png")
plt.close()

# =====================================================================
# CHART 2: ROC Curves
# =====================================================================
fpr_log, tpr_log, _ = roc_curve(y_test, y_score_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, color="#4A90E2", lw=2, label=f"Logistic Regression (AUC = {roc_auc_log:.3f})")
plt.plot(fpr_svm, tpr_svm, color="#E06D53", lw=2, label=f"Support Vector Machine (AUC = {roc_auc_svm:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=10)
plt.title("ROC Curves Comparison (Receiver Operating Characteristic)", fontsize=14, fontweight="bold", pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "9_roc_curves.png"), dpi=300)
print("Saved ROC Curves comparison to: 9_roc_curves.png")
plt.close()

# =====================================================================
# CHART 3: Feature Coefficients (Interpretability)
# =====================================================================
# Get coefficients
coef_log = log_reg.coef_[0]
coef_svm = svm.coef_[0]

coef_df = pd.DataFrame({
    'Feature': features * 2,
    'Coefficient Weight': np.concatenate([coef_log, coef_svm]),
    'Model': ['Logistic Regression'] * 6 + ['Support Vector Machine'] * 6
})

# Make the feature names clean for display
feature_mapping = {
    'is_rush_hour': 'Rush Hour Period',
    'direction_inbound': 'Inbound Direction',
    'adverse_weather_score': 'Weather Severity',
    'rush_weather_interaction': 'Rush x Weather Interaction',
    'dow_sin': 'Day of Week (Sin)',
    'dow_cos': 'Day of Week (Cos)'
}
coef_df['Feature'] = coef_df['Feature'].map(feature_mapping)

plt.figure(figsize=(10, 6))
# Plot side-by-side bar chart
ax = sns.barplot(x='Coefficient Weight', y='Feature', hue='Model', data=coef_df, palette=['#4A90E2', '#E06D53'])
plt.title('Model Coefficients Comparison (Feature Importance)', fontsize=14, fontweight="bold", pad=15)
plt.xlabel('Coefficient Value / Weight', fontsize=12, labelpad=10)
plt.ylabel('Features', fontsize=12, labelpad=10)
plt.legend(title='Classifier', loc='lower right')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Vertical line at 0 weight
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "10_feature_coefficients.png"), dpi=300)
print("Saved Model Coefficients comparison to: 10_feature_coefficients.png")
plt.close()
