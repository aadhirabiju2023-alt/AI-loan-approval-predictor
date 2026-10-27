import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = "data/loan_approval_dataset.csv"
df = pd.read_csv(file_path)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove extra spaces from text values
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# Convert text columns into numbers
df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Create input features (X)
X = df.drop(columns=["loan_id", "loan_status"])

# Create target/output (y)
y = df["loan_status"]

print("X and y created successfully!")

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())
# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split successfully!")

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nTraining target:")
print(y_train.shape)

print("\nTesting target:")
print(y_test.shape)
# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully!")

# Create the ML model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

print("\nPredictions made successfully!")

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save the trained model and scaler
import joblib

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully!")
# ==============================
# RANDOM FOREST MODEL
# ==============================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train Random Forest
random_forest.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")

# Make predictions
rf_pred = random_forest.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n===== RANDOM FOREST PERFORMANCE =====")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
# Save Random Forest model
joblib.dump(random_forest, "models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
# ==============================
# FEATURE IMPORTANCE
# ==============================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance.to_string(index=False))