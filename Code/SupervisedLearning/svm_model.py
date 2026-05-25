import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

# --- Custom Evaluation Functions ---
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def predict_svm(X, beta, beta_0):
    # Returns strictly 1 or -1 to prevent multiclass errors
    return np.where(beta_0 + np.dot(X, beta) >= 0, 1, -1)

def hinge_loss(X, y, beta, beta_0):
    margin = y * (beta_0 + np.dot(X, beta))
    return np.mean(np.maximum(0, 1 - margin))

# --- Load Data ---
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
# Strictly map target to {-1, 1} for mathematical SVM formulation
y = np.where(df[target].astype(int) == 0, -1, 1)

# Standardize numerical features
scaler = StandardScaler()
X_processed = scaler.fit_transform(X)

# Split 50% Train - 50% Test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.5, random_state=42, stratify=y
)

# --- Model Training ---
svm_model = LinearSVC(max_iter=10000, random_state=42)
svm_model.fit(X_train, y_train)

beta = svm_model.coef_[0]
beta_0 = svm_model.intercept_[0]

# --- Evaluation ---
y_train_pred = predict_svm(X_train, beta, beta_0)
y_test_pred = predict_svm(X_test, beta, beta_0)

train_loss = hinge_loss(X_train, y_train, beta, beta_0)
train_acc = accuracy(y_train, y_train_pred)
test_loss = hinge_loss(X_test, y_test, beta, beta_0)
test_acc = accuracy(y_test, y_test_pred)

# Advanced Metrics (pos_label=1 is standard for congestion)
test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_test_pred)

print("\n--- Advanced Confusion Matrix ---")
print(conf_matrix)

metrics_summary = pd.DataFrame({
    'Metric / Statistic': [
        'Training Set Shape', 'Testing Set Shape',
        'Training Hinge Loss', 'Training Accuracy',
        'Testing Hinge Loss', 'Testing Accuracy',
        'Testing Precision', 'Testing Recall', 'Testing F1-Score'
    ],
    'Value': [
        str(X_train.shape), str(X_test.shape),
        f"{train_loss:.4f}", f"{train_acc:.4f}",
        f"{test_loss:.4f}", f"{test_acc:.4f}",
        f"{test_precision:.4f}", f"{test_recall:.4f}", f"{test_f1:.4f}"
    ]
})

print("\n================ SUPPORT VECTOR MACHINE METRICS ================")
print(metrics_summary.to_string(index=False))
print("================================================================")
