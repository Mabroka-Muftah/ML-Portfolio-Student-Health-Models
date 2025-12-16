import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import sklearn

#-=================================================Enhancing the app UI========================================================================================
# Custom CSS for a clean, modern look

st.markdown("""
<style>
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Subheader */
    h2, h3 {
        color: #3498db;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    
    /* Success message */
    .stAlert[data-baseweb="alert"] {
        background-color: #e8f4f8;
        border: 1px solid #3498db;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)
#===============================================================================================================================================================


#Loading our trained models
Clf_model = jb.load("student_rf_model.pkl")
Reg_model =jb.load("cancer_rf_model.pkl")

#================================================================Students Data=====================================================================================

#Feature names Expected by model
ST_feature_names  = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Previous qualification (grade)', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP']

# Set default values for all features
ST_defaults = {
    'Marital status': 1.0, 'Application mode': 17.0, 'Application order': 1.0, 'Course': 9238.0, 'Daytime/evening attendance\t': 1.0, 'Previous qualification': 1.0, 'Previous qualification (grade)': 133.1, 'Nacionality': 1.0, "Mother's qualification": 19.0, "Father's qualification": 19.0, "Mother's occupation": 5.0, "Father's occupation": 7.0, 'Admission grade': 126.0, 'Displaced': 1.0, 'Educational special needs': 0.0, 'Debtor': 0.0, 'Tuition fees up to date': 1.0, 'Gender': 0.0, 'Scholarship holder': 0.0, 'Age at enrollment': 20.0, 'International': 0.0, 'Curricular units 1st sem (credited)': 0.0, 'Curricular units 1st sem (enrolled)': 6.0, 'Curricular units 1st sem (evaluations)': 8.0, 'Curricular units 1st sem (approved)': 5.0, 'Curricular units 1st sem (grade)': 12.32, 'Curricular units 1st sem (without evaluations)': 0.0, 'Curricular units 2nd sem (credited)': 0.0, 'Curricular units 2nd sem (enrolled)': 6.0, 'Curricular units 2nd sem (evaluations)': 8.0, 'Curricular units 2nd sem (approved)': 5.0, 'Curricular units 2nd sem (grade)': 12.2, 'Curricular units 2nd sem (without evaluations)': 0.0, 'Unemployment rate': 11.1, 'Inflation rate': 1.4, 'GDP': 0.32
}

MOTHER_OCCU_MAP = {
    "Unemployed": 0.0,
    "Student": 1.0,
    "Professional": 2.0,
    "Administrative staff": 3.0,
    "Service worker": 4.0,
    "Manual laborer": 5.0,
    "Other": 6.0
}


# Student Helping Functions
def get_std_inputs():
    inputs = {}

    # === Top Important Features ====
    st.subheader(" 📊 Key Academic Indicators ")
    inputs["Curricular units 1st sem (approved)"] = st.slider("1st Semester Units Passed", 0.0, 26.0, 5.0)
    inputs["Curricular units 1st sem (grade)"] = st.slider("1st Semester Avg Grade (0-20)", 0.0, 20.0, 12.32)
    inputs["Curricular units 2nd sem (approved)"] = st.slider("2nd Semester Units Passed", 0.0, 26.0, 5.0)
    inputs["Curricular units 2nd sem (evaluations)"] = st.slider("2nd Semester Assessments Taken", 0.0, 33.0, 8.0)
    inputs["Curricular units 2nd sem (grade)"] = st.slider("2nd Semester Avg Grade (0-20)", 0.0, 20.0, 12.2)
    inputs["Tuition fees up to date"] = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
    inputs["Debtor"] = st.selectbox("Is Student a Debtor?", ["No", "Yes"])
    inputs["Mother's occupation"] = st.selectbox(
        'Mother\'s Occupation', 
             ['Unemployed', 'Student','Professional', 'Administrative staff', 'Service worker', 'Manual laborer', 'Other'])
    inputs["Displaced"] = st.selectbox("Lives Away from Home?", ["No", "Yes"], index=0)
    inputs["Scholarship holder"] = st.selectbox("Scholarship Holder?", ["No", "Yes"], index=0)
    

    # === Optional Features (Hidden by defaults) =====
    with st.expander (" Edit Additional Student Info (Optoinal)") :
        st.markdown("Only change these if you know the actual values. Improves your prediction accuracy")

        # Personal Info
        st.markdown("### 👤 Personal Information")
        marital_map = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Legally Separated", 6: "Other"}
        inputs["Marital status"] = st.selectbox(
            "Marital Status", 
            list(marital_map.values()), 
            index=list(marital_map.keys()).index(1)  # Default: Single
        )
        gender_map = {0: "Female", 1: "Male"}
        inputs["Gender"] = st.selectbox("Gender", list(gender_map.values()), index=0)  # Default: Female
        inputs["Age at enrollment"] = st.number_input("Age at Enrollment", 16, 70, 20)
        inputs["International"] = st.selectbox("International Student?", ["No", "Yes"], index=0)
        inputs["Educational special needs"] = st.selectbox("Has Special Needs?", ["No", "Yes"], index=0)
        
        # Admission Information
        st.markdown("### 📝 Admission Information")
        inputs["Application mode"] = st.number_input("Application Mode (Code)", 1, 50, 17)
        inputs["Application order"] = st.number_input("Application Priority (1=Highest)", 1, 9, 1)
        inputs["Course"] = st.number_input("Course Code", 1000, 9999, 9238)
        inputs["Daytime/evening attendance\t"] = st.selectbox("Attendance Type", ["Daytime", "Evening"], index=0)  # 1=Daytime
        inputs["Previous qualification"] = st.number_input("Previous Qualification (Code)", 1, 40, 1)
        inputs["Previous qualification (grade)"] = st.number_input("Previous Qualification Grade (0-200)", 0.0, 200.0, 133.1)
        inputs["Admission grade"] = st.number_input("Admission Grade (0-20)", 0.0, 20.0, 12.6)

         # Parent Information
        st.markdown("### 👨‍👩‍👧 Parent Information")
        inputs["Nacionality"] = st.number_input("Nationality Code", 1, 200, 1)
        inputs["Mother's qualification"] = st.number_input("Mother's Qualification Code", 1, 40, 19)
        inputs["Father's qualification"] = st.number_input("Father's Qualification Code", 1, 40, 19)
        inputs["Father's occupation"] = st.number_input("Father's Occupation Code", 0, 10, 7)
        
        # Macroeconomic Context
        st.markdown("### 🌍 Macroeconomic Context (National)")
        inputs["Unemployment rate"] = st.number_input("Unemployment Rate (%)", 0.0, 30.0, 11.1)
        inputs["Inflation rate"] = st.number_input("Inflation Rate (%)", -5.0, 20.0, 1.4)
        inputs["GDP"] = st.number_input("GDP (per capita in 1000s)", 0.0, 2.0, 0.32)
        
        # Additional Curricular Details (rarely needed)
        st.markdown("### 📚 Additional Curricular Details")
        inputs["Curricular units 1st sem (credited)"] = st.number_input("1st Sem Credited Units", 0.0, 30.0, 0.0)
        inputs["Curricular units 1st sem (enrolled)"] = st.number_input("1st Sem Enrolled Units", 0.0, 30.0, 6.0)
        inputs["Curricular units 1st sem (without evaluations)"] = st.number_input("1st Sem Dropped Units", 0.0, 30.0, 0.0)
        inputs["Curricular units 2nd sem (credited)"] = st.number_input("2nd Sem Credited Units", 0.0, 30.0, 0.0)
        inputs["Curricular units 2nd sem (enrolled)"] = st.number_input("2nd Sem Enrolled Units", 0.0, 30.0, 6.0)
        inputs["Curricular units 2nd sem (without evaluations)"] = st.number_input("2nd Sem Dropped Units", 0.0, 30.0, 0.0)

    return inputs

def prepare_std_input(user_inputs, defaults):
    # Start with all defaults
    full_input = defaults.copy()
    
    # Override with user inputs
    for key, val in user_inputs.items():
        if key in full_input:
            # Handle Yes/No
            if val == "Yes":
                full_input[key] = 1.0
            elif val == "No":
                full_input[key] = 0.0
            # Handle Marital Status
            elif key == "Marital status":
                marital_map = {
                    "Single": 1.0, "Married": 2.0, "Widower": 3.0,
                    "Divorced": 4.0, "Legally Separated": 5.0, "Other": 6.0
                }
                full_input[key] = marital_map[val]
            # Handle Gender
            elif key == "Gender":
                full_input[key] = 0.0 if val == "Female" else 1.0
            # Handle Attendance
            elif key == "Daytime/evening attendance\t":
                full_input[key] = 1.0 if val == "Daytime" else 0.0
            elif key == "Mother's occupation":
                full_input[key] = MOTHER_OCCU_MAP[val]
            # Handle numeric inputs
            else:
                try:
                    full_input[key] = float(val)
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails
    
    # Convert to DataFrame (one row, all 36 features)
    return pd.DataFrame([full_input])


#=================================================================================================================================================================================================================================

# Streamlit App

st.set_page_config(
    page_title=" ML Portfolio: Student Model",
    layout="wide")
st.title("🎓 Student Academic Output Predictor 🎓")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>"
    "AI-Powered by Random Forest • Accuracy: 76%"
    "</p>",
    unsafe_allow_html=True
)

user_inputs = get_std_inputs()
if st.button("Predict Academic Outcome", key="student_btn"): #Critical: Without key, buttons in different tabs interfere
        input_df1 = prepare_std_input(user_inputs, ST_defaults)

        # make prediction
        pred = Clf_model.predict(input_df1)[0]
        proba = Clf_model.predict_proba(input_df1)[0]     

        # Get results
        
        labels = ["Dropout", "Enrolled", "Graduate"]
        st.markdown(
             f"""
              <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db;">
                 <h4 style="color: #2c3e50; margin: 0;">🎓 Prediction: {labels[pred]}</h4>
                 <p style="margin: 8px 0 0 0; color: #34495e;">Model confidence based on academic & socioeconomic profile</p>
              </div>
                """,
               unsafe_allow_html=True)
        for i, p in enumerate(proba):
         st.write(f"- {labels[i]}: {p:.1%}")

st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; color: #95a5a6; font-size: 14px; margin-top: 30px;">
        🌐 ML Portfolio Project • Random Forest Models • Data-Driven Decision Support
    </p>
    """,
    unsafe_allow_html=True
)

