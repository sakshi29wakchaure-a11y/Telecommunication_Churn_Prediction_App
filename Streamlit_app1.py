import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📡", layout="wide")

st.title("📡 Telecommunication Customer Churn Prediction")

# --- CACHING & DATA LOADING ---
@st.cache_data
def load_data():
    data_path = "c:/Users/HP/Desktop/Strimelit/churn_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "churn_dataset.csv" # fallback
    df = pd.read_csv(data_path)
    # Prepare data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_model():
    model_path = "c:/Users/HP/Desktop/Strimelit/models/churn_model_bundle.pkl"
    if not os.path.exists(model_path):
        model_path = "models/churn_model_bundle.pkl" # fallback
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

df = load_data()
bundle = load_model()
best_model = bundle["models"]["Random Forest"]

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
tab_selection = st.sidebar.radio("Go to", [
    "Data Overview",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Correlation Analysis",
    "ML Prediction"
])

# Define top-level KPIs for reuse
total_customers = df.shape[0]
total_churn = df[df['Churn'] == 'Yes'].shape[0] if 'Churn' in df.columns else df['Churn'].sum()
churn_rate = (total_churn / total_customers) * 100
avg_monthly_charges = df['MonthlyCharges'].mean()

# ---------------------------------------
# 1️⃣ DATA OVERVIEW TAB
# ---------------------------------------
if tab_selection == "Data Overview":
   
    
    # 🌟 KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Churn Customers", f"{total_churn:,}")
    col3.metric("Churn Rate %", f"{churn_rate:.2f}%")
    col4.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)
        
        st.subheader("Data Types & Missing Values")
        summary_df = pd.DataFrame({
            'Data Type': df.dtypes,
            'Missing Values': df.isnull().sum()
        })
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.subheader("First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # 📊 Charts Required
    st.markdown("### Numeric Column Analysis")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # Bar chart min max
        min_max_df = pd.DataFrame({
            'Min': df[num_cols].min(),
            'Max': df[num_cols].max()
        }).reset_index().melt(id_vars='index')

        fig = px.bar(min_max_df, x='index', y='value', color='variable', barmode='group',
                     title="Min and Max Values of All Numeric Columns",
                     labels={'index': 'Numeric Columns', 'value': 'Value'},
                     color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Numeric Column Distribution Summary**")
        st.dataframe(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

    # Business Insights
    st.success("**Business Insight:** The dataset captures holistic customer profiles including demographics, services signed up for, and account details. The 26.5% churn rate highlights a significant retention issue that warrants targeted intervention.")

# ---------------------------------------
# 2️⃣ UNIVARIATE ANALYSIS TAB
# ---------------------------------------
elif tab_selection == "Univariate Analysis":
    
    
    c1, c2 = st.columns(2)

    # Graph 1: Churn count plot
    with c1:
        st.metric(label="Churn %", value=f"{churn_rate:.1f}%")
        fig1 = px.histogram(df, x="Churn", color="Churn", title="1. Churn Count Plot", text_auto=True,
                            color_discrete_map={"Yes": "#ef553b", "No": "#636efa"})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Contract type
    with c2:
        top_contract = df['Contract'].mode()[0]
        st.metric(label="Most Common Contract", value=top_contract)
        fig2 = px.histogram(df, x="Contract", color="Contract", title="2. Contract Type Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    # Graph 3: Monthly Charges histogram
    with c3:
        st.metric(label="Average Monthly Charges", value=f"${avg_monthly_charges:.2f}")
        fig3 = px.histogram(df, x="MonthlyCharges", title="3. Monthly Charges Distribution", nbins=30, color_discrete_sequence=['#00cc96'])
        st.plotly_chart(fig3, use_container_width=True)

    # Graph 4: Tenure distribution
    with c4:
        avg_tenure = df['tenure'].mean()
        st.metric(label="Average Tenure", value=f"{avg_tenure:.1f} months")
        fig4 = px.histogram(df, x="tenure", title="4. Tenure Distribution", nbins=30, color_discrete_sequence=['#ab63fa'])
        st.plotly_chart(fig4, use_container_width=True)
        
    st.success("**Business Insight:** Most customers are on Month-to-Month contracts, representing the highest flight risk. We also observe a large spike in users with very low tenure (new customers), suggesting we need better onboarding to reduce early-stage churn.")

# ---------------------------------------
# 3️⃣ BIVARIATE ANALYSIS TAB
# ---------------------------------------
elif tab_selection == "Bivariate Analysis":
   

    # Helper function to compute churn % per group
    def get_churn_kpi(col_name):
        grouped = df.groupby(col_name)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        max_group = grouped.idxmax()
        max_val = grouped.max()
        return f"{max_group} ({max_val:.1f}%)"

    c1, c2 = st.columns(2)

    # Graph 1: Churn vs Contract
    with c1:
        st.metric("Highest Churn Contract", get_churn_kpi("Contract"))
        fig1 = px.histogram(df, x="Contract", color="Churn", barmode="group", text_auto=True,
                            title="1. Churn by Contract Type", color_discrete_map={"Yes": "#ef553b", "No": "#636efa"})
        st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Churn vs Internet Service
    with c2:
        st.metric("Highest Churn Internet Service", get_churn_kpi("InternetService"))
        fig2 = px.histogram(df, x="InternetService", color="Churn", barmode="group", text_auto=True,
                            title="2. Churn by Internet Service", color_discrete_map={"Yes": "#ef553b", "No": "#636efa"})
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    # Graph 3: Churn vs Payment Method
    with c3:
        st.metric("Highest Churn Payment Method", get_churn_kpi("PaymentMethod"))
        fig3 = px.histogram(df, x="PaymentMethod", color="Churn", barmode="group", text_auto=True,
                            title="3. Churn by Payment Method", color_discrete_map={"Yes": "#ef553b", "No": "#636efa"})
        st.plotly_chart(fig3, use_container_width=True)

    # Graph 4: Churn vs Monthly Charges (Boxplot)
    with c4:
        churn_avg_charges = df[df['Churn']=='Yes']['MonthlyCharges'].mean()
        non_churn_avg_charges = df[df['Churn']=='No']['MonthlyCharges'].mean()
        st.metric("Avg Charges (Churn vs No Churn)", f"${churn_avg_charges:.0f} vs ${non_churn_avg_charges:.0f}")
        fig4 = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                      title="4. Monthly Charges vs Churn", color_discrete_map={"Yes": "#ef553b", "No": "#636efa"})
        st.plotly_chart(fig4, use_container_width=True)

    st.success("**Business Insight:** Customers utilizing Fiber Optic internet, Month-to-month contracts, and Electronic checks have substantially higher churn rates. Higher Monthly Charges are also a strong predictor of churn. Offering mid-tier competitive pricing and locking them into 1-year contracts with incentives could mitigate this risk.")

# ---------------------------------------
# 4️⃣ CORRELATION ANALYSIS TAB
# ---------------------------------------
elif tab_selection == "Correlation Analysis":
 

    df_enc = df.copy()
    le = LabelEncoder()
    # Encoding categorical fields for correlation
    for col in df_enc.select_dtypes(include='object').columns:
        if col != 'customerID':
            df_enc[col] = le.fit_transform(df_enc[col])
            
    corr = df_enc.drop('customerID', axis=1, errors='ignore').corr()
    churn_corr = corr['Churn'].drop('Churn')
    
    most_pos_feature = churn_corr.idxmax()
    most_pos_val = churn_corr.max()
    most_neg_feature = churn_corr.idxmin()
    most_neg_val = churn_corr.min()

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Most Positive Correlated Feature", f"{most_pos_feature} ({most_pos_val:.2f})")
    c2.metric("Most Negative Correlated Feature", f"{most_neg_feature} ({most_neg_val:.2f})")
    c3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig)

    st.success("**Business Insight:** Tenure is strongly negatively correlated with churn, meaning loyal customers stick around. Monthly Charges are positively correlated, indicating price sensitivity. The contract type shows a strong relationship too, highlighting the importance of securing long-term commitments.")

# ---------------------------------------
# 5️⃣ ML PREDICTION TAB
# ---------------------------------------
elif tab_selection == "ML Prediction":
   
    
    # 1. Prepare Test Data for KPIs
    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
   
   
    
    # Generate predictions on test set to show real KPIs
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.markdown("---")
    st.subheader("🔮 Predict New Customer Churn")

    # User Input Form
    with st.form("prediction_form"):
        st.write("Enter Customer Details:")
        c1, c2 = st.columns(2)
        
        with c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            monthly = st.number_input("Monthly Charges", min_value=10.0, max_value=200.0, value=75.0)
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with c2:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

        submit = st.form_submit_button("Predict Churn")

    if submit:
        # Default values for hidden fields
        gender = "Male"
        senior = 0
        partner = "No"
        dependents = "No"
        phone = "Yes"
        multilines = "No"
        backup = "No"
        prot = "No"
        tv = "No"
        movies = "No"
        paperless = "Yes"
        
        # Create input dataframe matching exact columns during training
        total_charges = tenure * monthly
        
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multilines,
            "InternetService": internet,
            "OnlineSecurity": sec,
            "OnlineBackup": backup,
            "DeviceProtection": prot,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])
        
        # Make Prediction using the pipeline
        prob = best_model.predict_proba(input_data)[0][1]
        
        # Determine Risk Category and Business Recommendation
        if prob < 0.33:
            risk_category = "Low"
            recommendation = "Continue engaging them. Excellent opportunity for cross-selling deeper tech support or premium devices."
            color = "🟢"
        elif prob < 0.66:
            risk_category = "Medium"
            recommendation = "Monitor activity. Consider sending a satisfaction survey or offering a proactive discount."
            color = "🟡"
        else:
            risk_category = "High"
            recommendation = "Offer a discount, prioritize issue resolution, or propose an engaging downgrade path rather than losing them completely."
            color = "🔴"

        st.markdown("### Prediction Result")
        st.metric(label="Churn Probability", value=f"{prob:.2%}")
        
        st.write(f"**Risk Category:** {color} {risk_category}")
        st.info(f"💡 **Business Recommendation:** {recommendation}")

    st.success("**Business Insight:** Using this Random Forest model, the retention team can prioritize saving customers who have high churn probabilities (e.g., >75%). By shifting from reactive customer handling to proactive AI-driven engagement, the company can drastically reduce lost revenue.")