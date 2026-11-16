
import pandas as pd
import joblib
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Test Model
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

# Save Model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(iris.target_names, "models/flower_names.pkl")

# Save Results
results = {
    "accuracy": accuracy,
    "report": classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist()
}

joblib.dump(results, "models/results.pkl")

print("FloraSense AI Model Training Completed!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Model and results saved successfully.")
