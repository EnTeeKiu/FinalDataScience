import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import os

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function for Logistic Regression
def predict_logistic(X, beta, beta_0):
    probs = sigmoid(beta_0 + np.dot(X, beta))
    return (probs >= 0.5).astype(int)

# Logistic Loss / Binary Cross-Entropy
def logistic_loss(X, y, beta, beta_0):
    y_pred = sigmoid(beta_0 + np.dot(X, beta))

    # Avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(
        y * np.log(y_pred) +
        (1 - y) * np.log(1 - y_pred)
    )


file_path = os.path.join("Processed Datasets", "Cleaned Dataset.csv")
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please ensure the path is correct.")

df = pd.read_csv(file_path)

# Features and target
features = [
    'direction', 'is_weekend', 'hour_of_day', 'weather',
    'temp', 'rain_mm', 'humidity', 'visibility',
    'day_of_week', 'is_holiday'
]
target = 'is_congested'

X = df[features]

# Logistic Regression uses labels {0, 1}
y = df[target].astype(int)

# Preprocessing pipelines
categorical_cols = ['direction', 'weather', 'day_of_week']
numerical_cols = [
    'is_weekend', 'hour_of_day', 'temp',
    'rain_mm', 'humidity', 'visibility', 'is_holiday'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Splitting Data for Classification Task
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Convert y to numpy arrays to ensure compatibility with custom functions
y_train_cls = y_train_cls.values
y_test_cls = y_test_cls.values

# Printing shape of train and test sets
print("\nShape of training set (X_train_cls):", X_train_cls.shape)
print("Shape of testing set (X_test_cls):", X_test_cls.shape)

# Checking missing values
print("\nMissing values in the training set:")
print("X:", np.isnan(X_train_cls).sum(), "y:", np.isnan(y_train_cls).sum())

print("\nMissing values in the testing set:")
print("X:", np.isnan(X_test_cls).sum(), "y:", np.isnan(y_test_cls).sum())

# Count class distribution
train_counts = pd.Series(y_train_cls).value_counts()
test_counts = pd.Series(y_test_cls).value_counts()

print("\nTraining set class distribution:")
print(f"  Positive (1): {train_counts.get(1, 0)}")
print(f"  Negative (0): {train_counts.get(0, 0)}")

print("\nTest set class distribution:")
print(f"  Positive (1): {test_counts.get(1, 0)}")
print(f"  Negative (0): {test_counts.get(0, 0)}")

print("\nUnique values in y_train_cls:", np.unique(y_train_cls))
print("Unique values in y_test_cls:", np.unique(y_test_cls))

# Initialize Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)

# Fit model
log_model.fit(X_train_cls, y_train_cls)

# Extract parameters
beta = log_model.coef_[0]
beta_0 = log_model.intercept_[0]

# Print results
print("\nbeta:", beta)
print("beta_0:", beta_0)

# Training evaluation
train_loss = logistic_loss(X_train_cls, y_train_cls, beta, beta_0)
train_acc = accuracy(y_train_cls, predict_logistic(X_train_cls, beta, beta_0))

# Testing evaluation
test_loss = logistic_loss(X_test_cls, y_test_cls, beta, beta_0)
test_acc = accuracy(y_test_cls, predict_logistic(X_test_cls, beta, beta_0))

# Print results
print("\nLogistic Regression Results")
print("-" * 40)
print(f"Training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
print(f"Testing loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")