import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
import os

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Prediction function for SVM (sign function)
def predict_svm(X, beta, beta_0):
    return np.sign(beta_0 + np.dot(X, beta))

# Hinge Loss for SVM using X, y, beta, and beta_0
def hinge_loss(X, y, beta, beta_0):
    margin = y * (beta_0 + np.dot(X, beta))
    return np.mean(np.maximum(0, 1 - margin))

# File path setup to locate Cleaned Dataset.csv in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, "Cleaned Dataset.csv")
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

# Map target from {0, 1} to {-1, 1} for mathematical SVM formulation
y = df[target].map({0: -1, 1: 1}).astype(int)

# Preprocessing pipelines
categorical_cols = ['direction', 'weather', 'day_of_week']
numerical_cols = ['is_weekend', 'hour_of_day', 'temp', 'rain_mm', 'humidity', 'visibility', 'is_holiday']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Transform features (converts dataframe to numpy array which np.dot requires)
X_processed = preprocessor.fit_transform(X)

# Splitting Data for Classification Task
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Convert y to numpy arrays to ensure compatibility with custom functions
y_train_cls = y_train_cls.values
y_test_cls = y_test_cls.values

# Printing the shape of train and test sets
print("\nShape of training set (X_train_cls):", X_train_cls.shape)
print("Shape of testing set (X_test_cls):", X_test_cls.shape)

# Checking for missing values (Using np.isnan since X and y are numpy arrays now)
print("\nMissing values in the training set:")
print("X:", np.isnan(X_train_cls).sum(), "y:", np.isnan(y_train_cls).sum())
print("\nMissing values in the testing set:")
print("X:", np.isnan(X_test_cls).sum(), "y:", np.isnan(y_test_cls).sum())

# Count the number of positive and negative samples in training and test sets
train_counts = pd.Series(y_train_cls).value_counts()
test_counts = pd.Series(y_test_cls).value_counts()

# Print the counts
print("\nTraining set class distribution:")
print(f"  Positive (1): {train_counts.get(1, 0)}")
print(f"  Negative (-1): {train_counts.get(-1, 0)}")
print("\nTest set class distribution:")
print(f"  Positive (1): {test_counts.get(1, 0)}")
print(f"  Negative (-1): {test_counts.get(-1, 0)}")
print("\nUnique values in y_train_cls:", np.unique(y_train_cls))
print("Unique values in y_test_cls:", np.unique(y_test_cls))

# Initialize SVM model
svm_model = LinearSVC(max_iter=10000, random_state=42)

# Fit model
svm_model.fit(X_train_cls, y_train_cls)

# Extract parameters
beta = svm_model.coef_[0]
beta_0 = svm_model.intercept_[0]

# Print results
print("\nbeta:", beta)
print("beta_0:", beta_0)

# Training evaluation
train_loss = hinge_loss(X_train_cls, y_train_cls, beta, beta_0)
train_acc = accuracy(y_train_cls, predict_svm(X_train_cls, beta, beta_0))

# Testing evaluation
test_loss = hinge_loss(X_test_cls, y_test_cls, beta, beta_0)
test_acc = accuracy(y_test_cls, predict_svm(X_test_cls, beta, beta_0))

# Print results
print(f"\nTraining loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
print(f"Testing loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

# Create a metrics summary DataFrame to synthesize all statistics
metrics_summary = pd.DataFrame({
    'Metric / Statistic': [
        'Training Set Shape', 'Testing Set Shape',
        'Training Missing Values (X)', 'Training Missing Values (y)',
        'Testing Missing Values (X)', 'Testing Missing Values (y)',
        'Training Positive Class (1)', 'Training Negative Class (-1)',
        'Testing Positive Class (1)', 'Testing Negative Class (-1)',
        'Training Hinge Loss', 'Training Accuracy',
        'Testing Hinge Loss', 'Testing Accuracy',
        'Model Bias (beta_0)', 'Model Weights Count'
    ],
    'Value': [
        str(X_train_cls.shape), str(X_test_cls.shape),
        int(np.isnan(X_train_cls).sum()), int(np.isnan(y_train_cls).sum()),
        int(np.isnan(X_test_cls).sum()), int(np.isnan(y_test_cls).sum()),
        int(train_counts.get(1, 0)), int(train_counts.get(-1, 0)),
        int(test_counts.get(1, 0)), int(test_counts.get(-1, 0)),
        f"{train_loss:.4f}", f"{train_acc:.4f}",
        f"{test_loss:.4f}", f"{test_acc:.4f}",
        f"{beta_0:.4f}", len(beta)
    ]
})
print("\n================== METRICS SYNTHESIS TABLE ==================")
print(metrics_summary.to_string(index=False))
print("=============================================================")
