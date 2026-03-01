import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc
)


st.set_page_config(layout="wide", page_title="Churn EDA & Modeling")

# small style polish
st.markdown(
    """
    <style>
    .stApp { background-color: #f7fafc }
    .big-title { font-size:28px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    # fallback to workspace file
    try:
        return pd.read_csv("churn_dataset.csv")
    except Exception:
        # last resort: user's downloads
        default_path = r"C:\Users\HP\Downloads\churn_dataset.csv"
        try:
            return pd.read_csv(default_path)
        except Exception:
            return None


@st.cache_data
def preprocess(df_in):
    df = df_in.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna()
    return df


models_available = {
    "KNN": KNeighborsClassifier(),
    "Logistic": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}


@st.cache_data
def train_models(df, model_keys, test_size=0.25):
    data = df.copy()
    if "Churn" not in data.columns:
        raise ValueError("`Churn` column required for modeling")
    y = data["Churn"].map({"Yes":1, "No":0}) if data["Churn"].dtype == object else data["Churn"]
    X = data.drop(columns=["customerID", "Churn"], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse=False), cat_cols)
    ], remainder="drop")

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    for key in model_keys:
        model = models_available[key]
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs) if probs is not None else None
        pr_auc = average_precision_score(y_test, probs) if probs is not None else None

        results[key] = {
            "pipeline": pipe,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "preds": preds,
            "probs": probs,
            "y_test": y_test,
        }

    return results


def show_home(df):
    st.markdown("<div class='big-title'>Churn Analysis — Home</div>", unsafe_allow_html=True)
    st.write("A small interactive Streamlit app for exploring customer churn, training models, and making predictions.")
    st.markdown("---")
    st.subheader("Dataset sample & summary")
    st.dataframe(df.head())
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.write("Shape:", df.shape)
    with c2:
        st.write("Missing values:")
        st.write(df.isnull().sum())
    with c3:
        st.write("Describe (numeric):")
        st.write(df.describe())


def show_eda(df):
    st.header("Exploratory Data Analysis")
    tabs = st.tabs(["Univariate", "Bivariate", "Multivariate"])

    # Univariate
    with tabs[0]:
        st.subheader("Univariate")
        if "Churn" in df.columns:
            churn_counts = df["Churn"].value_counts()
            fig = px.pie(values=churn_counts.values, names=churn_counts.index, title="Churn distribution")
            st.plotly_chart(fig, use_container_width=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols[:6]:
            fig = px.histogram(df, x=c, nbins=40, title=f"{c} distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Bivariate
    with tabs[1]:
        st.subheader("Bivariate")
        if "Churn" in df.columns:
            for c in [x for x in num_cols if x not in ("customerID",)][:4]:
                fig = px.box(df, x="Churn", y=c, title=f"{c} by Churn")
                st.plotly_chart(fig, use_container_width=True)
        if "tenure" in df.columns:
            fig = px.scatter(df, x="tenure", y=num_cols[0] if num_cols else df.columns[0], color=(df["Churn"] if "Churn" in df.columns else None), title="Tenure vs other")
            st.plotly_chart(fig, use_container_width=True)

    # Multivariate
    with tabs[2]:
        st.subheader("Multivariate")
        st.write("Correlation heatmap (numeric features)")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


def show_prediction(df):
    st.header("Modeling & Prediction")
    st.write("Train models and predict churn for a single sample.")
    models = st.multiselect("Select models to train", list(models_available.keys()), default=["Logistic", "Random Forest"])
    test_pct = st.slider("Test set size (%)", 10, 40, 25)
    train_btn = st.button("Train")

    best_pipeline = None
    results = None
    if train_btn:
        try:
            results = train_models(df, models, test_size=test_pct/100.0)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return

        metrics_data = []
        for k, v in results.items():
            metrics_data.append({
                "Model": k,
                "Accuracy": v['accuracy'],
                "Precision": v['precision'],
                "Recall": v['recall'],
                "F1-Score": v['f1'],
            })
        st.dataframe(pd.DataFrame(metrics_data))

        acc_df = pd.DataFrame([{"model": k, "accuracy": v["accuracy"]} for k, v in results.items()])
        fig = px.bar(acc_df, x="model", y="accuracy", title="Model Accuracy")
        st.plotly_chart(fig, use_container_width=True)

        best_model_key = max(results.keys(), key=lambda k: results[k]["accuracy"]) if results else None
        best_pipeline = results[best_model_key]["pipeline"] if best_model_key else None
        st.success(f"Training complete. Best model: {best_model_key}")

    st.markdown("---")
    st.subheader("Predict single sample")
    if best_pipeline is None and results is None:
        st.info("Train a model first to enable single-sample prediction.")
    else:
        pipeline = best_pipeline if best_pipeline is not None else list(results.values())[0]['pipeline']
        X_cols = df.drop(columns=["customerID", "Churn"], errors="ignore")
        input_vals = {}
        with st.form("predict_form"):
            for c in X_cols.columns:
                if pd.api.types.is_numeric_dtype(X_cols[c]):
                    input_vals[c] = st.number_input(c, value=float(X_cols[c].median()))
                else:
                    unique = X_cols[c].dropna().unique().tolist()[:10]
                    input_vals[c] = st.selectbox(c, options=unique)
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([input_vals])
            pred_raw = pipeline.predict(row)[0]
            # map back to Yes/No if necessary
            pred_label = "Yes" if pred_raw == 1 or pred_raw == "Yes" else "No"
            prob = pipeline.predict_proba(row)[:,1][0] if hasattr(pipeline, "predict_proba") else None
            st.markdown(f"**Predicted churn:** {pred_label}")
            if prob is not None:
                st.progress(int(prob*100))
                st.write(f"Probability of churn: {prob:.3f}")


def main():
    st.sidebar.title("Navigation")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"] )
    page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction"] )

    df = load_data(uploaded)
    if df is None:
        st.warning("No dataset found. Upload a CSV or add `churn_dataset.csv` to the app folder.")
        st.stop()

    df = preprocess(df)

    if page == "Home":
        show_home(df)
    elif page == "EDA":
        show_eda(df)
    elif page == "Prediction":
        show_prediction(df)


if __name__ == '__main__':
    main()
