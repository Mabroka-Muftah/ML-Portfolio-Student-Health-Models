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
        background-color:  #2980b9;
        color: white;
        border: line;
        border-radius: 8px;
        padding: 10px 40px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        text-align: center;

    }
    
     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
       
</style>
""", unsafe_allow_html=True)
#===============================================================================================================================================================

#Loading our trained models
Reg_model =jb.load("cancer_rf_model.pkl")


#================================================================Cancer Data=====================================================================================

CA_feature_names = ['avganncount', 'avgdeathsperyear', 'incidencerate', 'medincome', 'popest2015', 'povertypercent', 'studypercap', 'medianage', 'medianagemale', 'medianagefemale', 'percentmarried', 'pctnohs18_24', 'pcths18_24', 'pctsomecol18_24', 'pctbachdeg18_24', 'pcths25_over', 'pctbachdeg25_over', 'pctemployed16_over', 'pctunemployed16_over', 'pctprivatecoverage', 'pctprivatecoveragealone', 'pctempprivcoverage', 'pctpubliccoverage', 'pctpubliccoveragealone', 'pctwhite', 'pctblack', 'pctasian', 'pctotherrace', 'pctmarriedhouseholds', 'birthrate']

CA_defaults = {'avganncount': 169.0, 'avgdeathsperyear': 61.0, 'incidencerate': 453.5494221, 'medincome': 45269.0, 'popest2015': 25788.0, 'povertypercent': 15.8, 'studypercap': 0.0, 'medianage': 41.0, 'medianagemale': 39.6, 'medianagefemale': 42.4, 'percentmarried': 52.5, 'pctnohs18_24': 17.2, 'pcths18_24': 34.8, 'pctsomecol18_24': 40.4, 'pctbachdeg18_24': 5.4, 'pcths25_over': 35.4, 'pctbachdeg25_over': 12.4, 'pctemployed16_over': 54.5, 'pctunemployed16_over': 7.6, 'pctprivatecoverage': 65.1, 'pctprivatecoveragealone': 48.7, 'pctempprivcoverage': 41.1, 'pctpubliccoverage': 36.4, 'pctpubliccoveragealone': 18.8, 'pctwhite': 90.12443712, 'pctblack': 2.231905108, 'pctasian': 0.543811087, 'pctotherrace': 0.844356882, 'pctmarriedhouseholds': 51.70068027, 'birthrate': 5.356186395}

# Cancer Helping Functions

def get_can_inputs():
    inputs = {}

    # Top 10 Important features
    with st.sidebar:
         st.subheader("📊 Key County Indicators")
         inputs["incidencerate"] = st.slider("Cancer incidence rate (cases per 100,000 people)",
                    0.0, 100000.0, 453.5494221)
         
         inputs["avgdeathsperyear"] = st.slider("Average annual number of cancer deaths",
             0.0, 100000.0, 61.0 )
         
         inputs["medincome"] = st.number_input("Median Household Income ($)", 
              min_value=0.0,max_value=1000000.0,value=45269.0,step=1000.0, )
         
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
              min_value=0.0, max_value=40000000.0, value=25788.0, step=1000.0  )
             inputs["studypercap"] = st.number_input(
                       "Cancer studies per capita", 
                        min_value=0.0, max_value=10000.0, value=0.0, step=1.0 )
    
             # Age Metrics
             st.markdown("### 👵 Age Demographics")
       
             inputs["medianage"] = st.number_input(
                "Median age (total population)", 
                     min_value=0.0, max_value=100.0, value=41.0, step=0.1  )
             inputs["medianagemale"] = st.number_input(  "Median age (males)", 
                      min_value=0.0, max_value=100.0, value=39.6, step=0.1
                                                    )
             inputs["medianagefemale"] = st.number_input(  "Median age (females)", 
                      min_value=0.0, max_value=100.0, value=42.4, step=0.1
               )
    
               # Family & Marital
             st.markdown("### 👨‍👩‍👧 Family Structure")
             inputs["percentmarried"] = st.slider( "% of adults who are married", 
               0.0, 100.0, 52.5)
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

         full_inputs[key] = val

    return pd.DataFrame([full_inputs])


#=================================================================================================================================================================================================================================

# Streamlit App

st.set_page_config(
    page_title=" ML Portfolio: Health Care prediction",
    layout="wide")
st.title("🏥 Cancer Mortality Predictor 🏥")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d;'>"
    "AI-Powered by Random Forest • R²: 0.55"
    "</p>",
    unsafe_allow_html=True
)

user_input = get_can_inputs()

if st.button("Predict Death Rate", key="cancer_btn"):
        input_df2 = prepare_can_inputs(user_input, CA_defaults)

        # Make prediction
        pred = int(Reg_model.predict(input_df2)[0]) # you cannot have a half person

        # get results
        st.markdown(
             f"""
              <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db;">
                 <h4 style="color: #2c3e50; margin: 0;">Predicted Death Rate: {pred} per 100,000 people annually</h4>
                 <p style="margin: 8px 0 0 0; color: #34495e;">Model confidence based on academic & socioeconomic profile</p>
              </div>
                """,
               unsafe_allow_html=True)
        st.info("Model Performance | Mean Error = ±14.1 deaths/100k")
st.image("dataset-cover (1).jpg", caption = "Cancer Death Rates Prediction Regression", width=1250)



