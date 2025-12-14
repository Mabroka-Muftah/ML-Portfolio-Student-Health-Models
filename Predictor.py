import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import sklearn

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
    st.subheader(" 📊 Key Academic Indicators")
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
        inputs["Displaced"] = st.selectbox("Lives Away from Home?", ["No", "Yes"], index=0)
        inputs["Scholarship holder"] = st.selectbox("Scholarship Holder?", ["No", "Yes"], index=0)
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



#================================================================Cancer Data=====================================================================================

CA_feature_names = ['avganncount', 'avgdeathsperyear', 'incidencerate', 'medincome', 'popest2015', 'povertypercent', 'studypercap', 'medianage', 'medianagemale', 'medianagefemale', 'percentmarried', 'pctnohs18_24', 'pcths18_24', 'pctsomecol18_24', 'pctbachdeg18_24', 'pcths25_over', 'pctbachdeg25_over', 'pctemployed16_over', 'pctunemployed16_over', 'pctprivatecoverage', 'pctprivatecoveragealone', 'pctempprivcoverage', 'pctpubliccoverage', 'pctpubliccoveragealone', 'pctwhite', 'pctblack', 'pctasian', 'pctotherrace', 'pctmarriedhouseholds', 'birthrate']

CA_defaults = {'avganncount': 169.0, 'avgdeathsperyear': 61.0, 'incidencerate': 453.5494221, 'medincome': 45269.0, 'popest2015': 25788.0, 'povertypercent': 15.8, 'studypercap': 0.0, 'medianage': 41.0, 'medianagemale': 39.6, 'medianagefemale': 42.4, 'percentmarried': 52.5, 'pctnohs18_24': 17.2, 'pcths18_24': 34.8, 'pctsomecol18_24': 40.4, 'pctbachdeg18_24': 5.4, 'pcths25_over': 35.4, 'pctbachdeg25_over': 12.4, 'pctemployed16_over': 54.5, 'pctunemployed16_over': 7.6, 'pctprivatecoverage': 65.1, 'pctprivatecoveragealone': 48.7, 'pctempprivcoverage': 41.1, 'pctpubliccoverage': 36.4, 'pctpubliccoveragealone': 18.8, 'pctwhite': 90.12443712, 'pctblack': 2.231905108, 'pctasian': 0.543811087, 'pctotherrace': 0.844356882, 'pctmarriedhouseholds': 51.70068027, 'birthrate': 5.356186395}

# Cancer Helping Functions

def get_can_inputs():
    inputs = {}

    # Top 10 Important features

    st.subheader("📊 Key County Indicators")
    inputs["incidencerate"] = st.slider("Cancer incidence rate (cases per 100,000 people)",
     0.0, 100000.0, 453.5494221)
    inputs["avgdeathsperyear"] = st.slider("Average annual number of cancer deaths",
    0.0, 100000.0, 61.0 )
    inputs["medincome"] = st.number_input("Median Household Income ($)", 
    min_value=0.0,
    max_value=1000000.0,
    value=45269.0,
    step=1000.0, )
    inputs["povertypercent"] = st.slider("% of population below poverty line", 0.0, 100.0, 15.8,  help="Percentage of population below federal poverty line")
    inputs["pctotherrace"] = st.slider("% of Other races ", 0.0, 100.0, 0.844356882)
    inputs["pctunemployed16_over"] = st.slider("% of 16+ year olds who are unemployed", 0.0, 100.0, 7.6)
    inputs["pcths25_over"] = st.slider("% of 25+ year olds with high school diploma (only)", 0.0, 100.0, 35.4)
    inputs["pctbachdeg25_over"] = st.slider("% of 25+ year olds with bachelor’s degree", 0.0, 100.0, 12.4)
    inputs["pctpubliccoveragealone"] = st.slider("% with only public insurance (no private)", 0.0, 100.0, 18.8)
    inputs["pctprivatecoverage"] = st.slider("% with any private insurance (employer, purchased, etc.)", 0.0, 100.0, 65.1)
    

    # Other Optional Features

    with st.expander("Edit Other County Info (Optional)"):
       st.markdown("Only change these if you know the actual values. Defaults will be used otherwise.")
    
        # Population & Counts
       st.markdown("### 📊 Population & Cancer Metrics")
       inputs["avganncount"] = st.number_input(
        "Average annual cancer cases", 
        min_value=0.0, max_value=100000.0, value=169.0, step=100.0
       )
       inputs["popest2015"] = st.number_input(
        "Population estimate (2015)", 
        min_value=0.0, max_value=40000000.0, value=25788.0, step=1000.0
        )
       inputs["studypercap"] = st.number_input(
        "Cancer studies per capita", 
        min_value=0.0, max_value=10000.0, value=0.0, step=1.0
       )
    
       # Age Metrics
       st.markdown("### 👵 Age Demographics")
       inputs["medianage"] = st.number_input(
        "Median age (total population)", 
        min_value=0.0, max_value=100.0, value=41.0, step=0.1
          )
       inputs["medianagemale"] = st.number_input(
        "Median age (males)", 
        min_value=0.0, max_value=100.0, value=39.6, step=0.1
       )
       inputs["medianagefemale"] = st.number_input(
        "Median age (females)", 
        min_value=0.0, max_value=100.0, value=42.4, step=0.1
       )
    
        # Family & Marital
       st.markdown("### 👨‍👩‍👧 Family Structure")
       inputs["percentmarried"] = st.slider(
        "% of adults who are married", 
        0.0, 100.0, 52.5
       )
       inputs["pctmarriedhouseholds"] = st.slider(
        "% of households headed by married couples", 
        0.0, 100.0, 51.70068027
       )
       inputs["birthrate"] = st.number_input(
        "Births per 1,000 people per year", 
        min_value=0.0, max_value=1000.0, value=5.356186395, step=0.1
       )
    
       # Education (18-24)
       st.markdown("### 🎓 Education (Ages 18-24)")
       inputs["pctnohs18_24"] = st.slider(
        "% with no high school diploma (18-24)", 
        0.0, 100.0, 17.2
       )
       inputs["pcths18_24"] = st.slider(
        "% with high school diploma only (18-24)", 
        0.0, 100.0, 34.8
       )
       inputs["pctsomecol18_24"] = st.slider(
        "% with some college (18-24)", 
        0.0, 100.0, 40.4
       )
       inputs["pctbachdeg18_24"] = st.slider(
        "% with bachelor's degree (18-24)", 
        0.0, 100.0, 5.4
       )
    
       # Employment
       st.markdown("### 💼 Employment")
       inputs["pctemployed16_over"] = st.slider(
        "% employed (16+ years)", 
        0.0, 100.0, 54.5
       )
    
       # Insurance (Additional)
       st.markdown("### 🏥 Health Insurance (Additional)")
       inputs["pctprivatecoveragealone"] = st.slider(
        "% with only private insurance", 
        0.0, 100.0, 48.7
       )
       inputs["pctempprivcoverage"] = st.slider(
        "% with employer-provided private insurance", 
        0.0, 100.0, 41.1
       )
       inputs["pctpubliccoverage"] = st.slider(
        "% with any public insurance (Medicaid/Medicare)", 
        0.0, 100.0, 36.4
       )
    
       # Race/Ethnicity
       st.markdown("### 🌍 Race/Ethnicity")
       inputs["pctwhite"] = st.slider(
        "% White population", 
        0.0, 100.0, 90.12443712
        )
       inputs["pctblack"] = st.slider(
        "% Black population", 
        0.0, 100.0, 2.231905108
       )
       inputs["pctasian"] = st.slider(
        "% Asian population", 
        0.0, 100.0, 0.543811087
       )

    return inputs

def prepare_can_inputs(user_inputs, defaults):

    # start with defaults
    full_inputs = defaults.copy()
    for key, val in user_inputs.items():
     if key in full_inputs:
        try:
         full_inputs[key] = float(val)
        except (ValueError, TypeError):
          pass

     return pd.DataFrame([full_inputs])


#=================================================================================================================================================================================================================================

# Streamlit App

st.set_page_config(
    page_title=" ML Portfolio: Student & Health Models",
    layout="wide")
st.title("🎓 Student Academic Performance  &  🏫 Public Health Predictors ")
st.markdown("Powered by Random Forest • Accuracy: 76% • R²: 0.55")

# separated tabs
tab1, tab2 = st.tabs(["🎓 Student Success ", "🏥 Cancer Mortality "])

# First tab: Students outcome prediction
with tab1:
    user_inputs = get_std_inputs()
    if st.button("Predict Academic Outcome", key="student_btn"): #Critical: Without key, buttons in different tabs interfere
        input_df1 = prepare_std_input(user_inputs, ST_defaults)

        # make prediction
        pred = Clf_model.predict(input_df1)[0]
        proba = Clf_model.predict_proba(input_df1)[0]     

        # Get results
        labels = ["Dropout", "Enrolled", "Graduate"]
        st.success(f"**Prediction**: {labels[pred]}")
        st.write("Confidence:")
        for i, p in enumerate(proba):
          st.write(f"- {labels[i]}: {p:.1%}")


# Second Tab: Cancer mortality prediciton
with tab2:
    user_input = get_can_inputs()
    if st.button("Predict Death Rate", key="cancer_btn"):
        input_df2 = prepare_can_inputs(user_input, CA_defaults)

        # Make prediction
        pred = int(Reg_model.predict(input_df2)[0]) # you cannot have a half person

        # get results
        st.success(f"**Predicted Death Rate**: {pred} per 100,000 people")
        st.info("Model Performance: R² = 0.55 | Mean Error = ±14.1 deaths/100k")



