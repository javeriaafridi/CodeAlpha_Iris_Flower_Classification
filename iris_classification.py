"""
CodeAlpha - Data Science Internship
Task 1: Iris Flower Classification

Trains and compares multiple classifiers to predict Iris species
(setosa, versicolor, virginica) from sepal/petal measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
OUT_DIR = "outputs"

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])  # drop row-id column

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass balance:\n", df["Species"].value_counts())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------
sns.pairplot(df, hue="Species", corner=True)
plt.savefig(f"{OUT_DIR}/pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.drop(columns=["Species"]).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
X = df.drop(columns=["Species"])
y = df["Species"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Train & compare multiple models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVM (RBF kernel)": SVC(kernel="rbf"),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

# ---------------------------------------------------------
# 5. Pick best model, show confusion matrix
# ---------------------------------------------------------
best_name = max(results, key=results.get)
best_model = models[best_name]
print(f"\nBest model: {best_name} (accuracy={results[best_name]:.4f})")

preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# Model comparison bar chart
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), results.values(), color="teal")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=20, ha="right")
plt.ylim(0.8, 1.01)
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll plots saved to outputs/")
