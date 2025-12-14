import streamlit as st
import joblib
import pandas as pd
import sklearn

# ==============================
# LOAD MODELS & DEFAULTS
# ==============================
@st.cache_resource
def load_assets():
    return {
        "student_model": joblib.load("student_rf_model.pkl"),
        "cancer_model": joblib.load("cancer_rf_model.pkl"),
        "student_defaults": joblib.load("student_all_defaults.pkl"),
        "cancer_defaults": joblib.load("cancer_all_defaults.pkl"),
        "top_student": joblib.load("top_student_features.pkl"),
        "top_cancer": joblib.load("top10_cancer_features.pkl")
    }

assets = load_assets()

student_model = assets["student_model"]
cancer_model = assets["cancer_model"]
student_defaults = assets["student_defaults"]
cancer_defaults = assets["cancer_defaults"]
TOP_STUDENT = assets["top_student"]
TOP_CANCER = assets["top_cancer"]

# ==============================
# HELPER FUNCTIONS
# ==============================
def create_full_input(user_inputs, defaults_dict):
    """Merge user inputs with defaults to create full feature vector."""
    full_input = defaults_dict.copy()
    for feat, val in user_inputs.items():
        if feat in full_input:
            full_input[feat] = val
    return pd.DataFrame([full_input])

def encode_yes_no(df):
    """Convert 'Yes'/'No' to 1/0 for model compatibility."""
    return df.replace({"Yes": 1, "No": 0})

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(
    page_title="🎓 ML Portfolio: Student & Health Models",
    layout="centered"
)

st.title("🎓🏫 Student Academic Success  & Cancer Regression Predictor")
st.markdown("Powered by Random Forest • Accuracy: 76% • R²: 0.55")

tab1, tab2 = st.tabs(["🎓 Student Success", "🏥 Cancer Risk"])

# ------------------------------
# TAB 1: STUDENT CLASSIFICATION
# ------------------------------
with tab1:
    st.subheader("Predict Student Academic Outcome")
    st.write("Adjust key factors to see the predicted result:")
    
    # Only show top features (limit to 6 for clean UI)
    top_feats = TOP_STUDENT[:8]
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    with col1:
        for feat in top_feats[:3]:
            if "grade" in feat.lower() or "units" in feat.lower():
                user_inputs[feat] = st.slider(f"{feat}", 0.0, 20.0, float(student_defaults[feat]))
            elif "age" in feat.lower():
                user_inputs[feat] = st.slider(f"{feat}", 16, 60, int(student_defaults[feat]))
            else:
                user_inputs[feat] = st.number_input(f"{feat}", value=float(student_defaults[feat]))
    
    with col2:
        for feat in top_feats[3:]:
            if feat in ["Tuition fees up to date", "Scholarship holder", "Displaced", "Debtor", "Gender", "International"]:
                user_inputs[feat] = st.selectbox(f"{feat}", ["Yes", "No"], 
                                                index=0 if student_defaults[feat] == 1 else 1)
            else:
                user_inputs[feat] = st.number_input(f"{feat}", value=float(student_defaults[feat]))
    
    if st.button("Predict Student Outcome"):
        # Build full input (all 36 features)
        full_df = create_full_input(user_inputs, student_defaults)
        full_df = encode_yes_no(full_df)
        
        # Predict
        pred = student_model.predict(full_df)[0]
        proba = student_model.predict_proba(full_df)[0]
        labels = ["Dropout", "Enrolled", "Graduate"]
        
        st.success(f"**Prediction**: {labels[pred]}")
        st.write("Confidence:")
        for i, p in enumerate(proba):
            st.write(f"- {labels[i]}: {p:.1%}")

# ------------------------------
# TAB 2: CANCER REGRESSION
# ------------------------------
with tab2:
    st.subheader("Estimate County Cancer Death Rate")
    st.write("Enter key demographic indicators (per 100,000 people):")
    
    # Only show top features (limit to 6)
    top_feats = TOP_CANCER[:10]
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    with col1:
        for feat in top_feats[:3]:
            if "income" in feat.lower():
                user_inputs[feat] = st.number_input(f"{feat} ($)", 30000, 150000, int(cancer_defaults[feat]))
            elif "age" in feat.lower():
                user_inputs[feat] = st.slider(f"{feat}", 25.0, 50.0, float(cancer_defaults[feat]))
            else:
                user_inputs[feat] = st.slider(f"{feat} (%)", 0.0, 100.0, float(cancer_defaults[feat]))
    
    with col2:
        for feat in top_feats[3:]:
            if "income" in feat.lower():
                user_inputs[feat] = st.number_input(f"{feat} ($)", 30000, 150000, int(cancer_defaults[feat]))
            elif "age" in feat.lower():
                user_inputs[feat] = st.slider(f"{feat}", 25.0, 50.0, float(cancer_defaults[feat]))
            else:
                user_inputs[feat] = st.slider(f"{feat} (%)", 0.0, 100.0, float(cancer_defaults[feat]))
    
    if st.button("Predict Death Rate"):
        # Build full input (all 32 features)
        full_df = create_full_input(user_inputs, cancer_defaults)
        
        # Predict
        pred = cancer_model.predict(full_df)[0]
        st.success(f"**Predicted Death Rate**: {pred:.1f} per 100,000 people")
        st.info("Model Performance: R² = 0.55 | Mean Absolute Error = ±14.1 deaths/100k")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("ML Portfolio Project • Random Forest Models • Data-Driven Insights")