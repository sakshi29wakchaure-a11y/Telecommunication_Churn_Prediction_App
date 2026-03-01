"""
save_models.py
--------------
Run this script ONCE to train the 4 notebook models and save them as a pickle bundle:

    python save_models.py

Output:
    models/churn_model_bundle.pkl   — dict with all 4 pipelines + metadata

The Streamlit app loads this bundle instantly on startup.

Note: KNN is excluded — it errored in the notebook and is too slow for this dataset size.
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. Load & clean data  (same as notebook)
# --------------------------------------------------
DATA_PATH = r"C:\Users\HP\Downloads\churn_dataset.csv"

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Fix TotalCharges (stored as object in original CSV)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# --------------------------------------------------
# 2. Preprocessing pipeline  (same as notebook)
# --------------------------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

print(f"Numeric columns  : {num_cols}")
print(f"Categorical cols : {cat_cols}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

# --------------------------------------------------
# 3. Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}\n")

# --------------------------------------------------
# 4. Define 4 models (KNN excluded — errored in notebook)
# --------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ]),
    "SVM": Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(probability=True, kernel="rbf")),
    ]),
    "Decision Tree": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=10, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            max_features="sqrt",
            max_samples=0.5,
            n_jobs=1,
            random_state=42,
        )),
    ]),
}

# --------------------------------------------------
# 5. Train each model and report accuracy
# --------------------------------------------------
print("Training models...\n")
trained = {}
for name, pipe in models.items():
    print(f"  [{name}]  training...", end=" ", flush=True)
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    trained[name] = pipe
    print(f"Accuracy = {acc:.4f}  DONE")

# --------------------------------------------------
# 6. Save bundle
# --------------------------------------------------
os.makedirs("models", exist_ok=True)
bundle = {
    "models": trained,
    "feature_columns": X.columns.tolist(),
    "num_cols": num_cols,
    "cat_cols": cat_cols,
}

out_path = os.path.join("models", "churn_model_bundle.pkl")
with open(out_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"\n✅  Saved → {out_path}")
print(f"   Models : {list(trained.keys())}")
