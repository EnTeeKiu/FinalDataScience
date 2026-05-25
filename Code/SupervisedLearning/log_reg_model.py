import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os

# --- Custom Evaluation Functions ---
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic(X, beta, beta_0):
    probs = sigmoid(beta_0 + np.dot(X, beta))
    return (probs >= 0.5).astype(int)

def logistic_loss(X, y, beta, beta_0):
    y_pred = sigmoid(beta_0 + np.dot(X, beta))
    eps = 1e-15 # Avoid log(0) numerical instability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# --- Load Data ---
file_path = os.path.join("Processed Datasets", "Processed Dataset.csv")
if not os.path.exists(file_path):
    file_path = os.path.join("..", "..", "Processed Datasets", "Processed Dataset.csv")

df = pd.read_csv(file_path)

# The 6 strictly selected features
features = [
    'is_rush_hour', 'direction_inbound', 'adverse_weather_score',
    'rush_weather_interaction', 'dow_sin', 'dow_cos'
]
target = 'is_congested'

X = df[features]
y = df[target].astype(int)

# Split 50% Train - 50% Test first to prevent data leakage, as strictly required by the Rubric. stratify=y handles class imbalance.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Standardize features AFTER splitting to prevent data leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

y_train_arr = y_train.values
y_test_arr = y_test.values

# --- Model Training ---
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train_arr)

beta = log_model.coef_[0]
beta_0 = log_model.intercept_[0]

# --- Evaluation ---
y_train_pred = predict_logistic(X_train, beta, beta_0)
y_test_pred = predict_logistic(X_test, beta, beta_0)

train_loss = logistic_loss(X_train, y_train_arr, beta, beta_0)
train_acc = accuracy(y_train_arr, y_train_pred)
test_loss = logistic_loss(X_test, y_test_arr, beta, beta_0)
test_acc = accuracy(y_test_arr, y_test_pred)

# Advanced Metrics for Imbalanced Traffic Data
test_precision = precision_score(y_test_arr, y_test_pred, zero_division=0)
test_recall = recall_score(y_test_arr, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test_arr, y_test_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test_arr, y_test_pred)

print("\n--- Advanced Confusion Matrix ---")
print(conf_matrix)

metrics_table = pd.DataFrame({
    "Metric / Statistic": [
        "Training Set Shape", "Testing Set Shape",
        "Training Logistic Loss", "Training Accuracy",
        "Testing Logistic Loss", "Testing Accuracy",
        "Testing Precision", "Testing Recall", "Testing F1-Score"
    ],
    "Value": [
        str(X_train.shape), str(X_test.shape),
        round(train_loss, 4), round(train_acc, 4),
        round(test_loss, 4), round(test_acc, 4),
        round(test_precision, 4), round(test_recall, 4), round(test_f1, 4)
    ]
})

print("\n================ LOGISTIC REGRESSION METRICS ================")
print(metrics_table.to_string(index=False))
print("=============================================================")