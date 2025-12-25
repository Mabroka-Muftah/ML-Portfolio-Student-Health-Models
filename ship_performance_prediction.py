import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved objects
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature lists (update these to match your 29 columns!)
numerical_features = [
    'Speed_Over_Ground_knots', 'Engine_Power_kW', 'Distance_Traveled_nm',
    'Draft_meters', 'Cargo_Weight_tons', 'Operational_Cost_USD',
    'Revenue_per_Voyage_USD', 'Turnaround_Time_hours',
    'Efficiency_nm_per_kWh', 'Seasonal_Impact_Score',
    'Weekly_Voyage_Count', 'Average_Load_Percentage'
]

categorical_features = {
    'Ship_Type': ['Bulk', 'Container', 'Fish', 'Tanker'],
    'Engine_Type': ['Diesel', 'HFO', 'Steam Turbine'],
    'Maintenance_Status': ['Good', 'Fair', 'Critical'],
    'Route_Type': ['Coastal', 'Transoceanic', 'Long haul', 'short haul'],
    'Weather_Condition': ['Calm', 'Moderate', 'Rough']
}

all_onehot_features = [
 'Speed_Over_Ground_knots',    #سرعة السفينة على الارض بالميل البحري في الساعة
 'Engine_Power_kW',            #انتاج الطاقة من المحرك الرئيسي اثناء الرحلة بالكيلووات
 'Distance_Traveled_nm',       #اجمالي الاميال البحرية التي سيتم اجتيازهااثناءالرحلة     nm = nautical mile
 'Draft_meters',               #العمق الرأسي لهيكل السفينة تحت الماء
 'Cargo_Weight_tons',          #الوزن الاجمالي للبضائع المنقولة
 'Operational_Cost_USD',       #التكلفة الاجمالية لتشغيل السفينة لهذه الرحلة
 'Revenue_per_Voyage_USD',     #الدخل المتولد من الرحلة(عقود الشحن) 
 'Turnaround_Time_hours',      #الوقت التس تقضيه في الميناء للشحن/التفريغ والتزود بالوقود
 'Efficiency_nm_per_kWh',      #مقياس الاداء الرئيسي, الاميال البحرية المقطوعة لكل كيلووات/ساعة من الطاقة
 'Seasonal_Impact_Score',      #مقدار الموسمية التي اثرت على العمليات درجة مشتقة من 1-9
 'Weekly_Voyage_Count',        #عدد الرحلات التي تكملها السفينة عادة في الاسبوع
 'Average_Load_Percentage',    #مدى امتلاء السفينة كنسبة مئوية من السعة الكلية
 'Ship_Type_Bulk Carrier',     #تصميم السفينة نقل سائب
 'Ship_Type_Container Ship',   #تصميم السفينة سفينة حاويات
 'Ship_Type_Fish Carrier',     #تصميم السفينة نقل سمك 
 'Ship_Type_Tanker',           #تصميم السفينة ناقلة
 'Route_Type_Coastal',         #طبيعة طريص الشحن : ساحلي
 'Route_Type_Long-haul',       #طبيعة طريص الشحن :مدى طويل 
 'Route_Type_Short-haul',      #طبيعة طريص الشحن : مدى قصير
 'Route_Type_Transoceanic',    #طبيعة طريص الشحن : عبر المحيط
 'Engine_Type_Diesel',         # نوع نظام الدفع: ديزل
 'Engine_Type_Heavy Fuel Oil (HFO)', # نوع نظام الدفع: زيت نقل ثقيل
 'Engine_Type_Steam Turbine',  # نوع نظام الدفع:توربين بخاري
 'Maintenance_Status_Critical', #حالة الصيانة: حرجة
 'Maintenance_Status_Fair',     #حالة الصيانة:  مُعرض
 'Maintenance_Status_Good',     #حالة الصيانة: جيدة
 'Weather_Condition_Calm',      
 'Weather_Condition_Moderate',
 'Weather_Condition_Rough']

#cluster labels
cluster_labels = {
    0: "High-Cost Carriers",
    1: "Cost-Efficient Carriers",
    2: "Specialized Vessels"
}

# Descriptions for users
cluster_descriptions = {
    "High-Cost Carriers": (
        "Vessels with higher operational costs and critical maintenance needs. "
        "Often older bulk/tanker ships using Heavy Fuel Oil (HFO). Consider efficiency upgrades."
    ),
    "Cost-Efficient Carriers": (
        "Modern, well-maintained ships (often bulk/container) with diesel engines. "
        "Lowest operational cost and reliable performance — ideal for standard voyages."
    ),
    "Specialized Vessels": (
        "Typically fishing or niche vessels (e.g., steam-powered). "
        "Well-maintained but technologically distinct. Best for specialized operations, not general cargo."
    )
}

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Ship Cluster Predictor", page_icon="🚢", layout='wide')
st.title("🚢 Ship Performance Predictor 🚢")
st.write("Enter ship details to identify its operational group.")


col1,col2, col3 = st.columns(3)

# Numerical inputs
with col1:  
 st.subheader("📊 Numerical Indicators")
 input_data = {}
 for feat in numerical_features:
    if feat == 'Efficiency_nm_per_kWh':
        val = st.number_input(feat, min_value=0.100211333, max_value=1.499259399, value=0.79865557, step=0.01, help='الاميال البحرية المقطوعة لكل كيلووات/ساعة من الطاقة')
    elif feat == 'Cargo_Weight_tons':
        val = st.number_input(feat, min_value=50.22962415, max_value=1999.126697, value=1032.573264, step=100.0, help="الوزن الاجمالي للبضائع المنقولة")
    elif feat == 'Operational_Cost_USD':
        val = st.number_input(feat, min_value=10092.30632, max_value=499734.8679, value=255143.3445, step=1000.0, help="التكلفة الاجمالية لتشغيل السفينة لهذه الرحلة")
    elif feat == 'Speed_Over_Ground_knots':
        val = st.number_input(feat, min_value =10.00975574 , max_value =24.99704335 ,value =17.50339954 , step =1.0 , help="سرعة السفينة على الارض بالميل البحري في الساعة")
    elif feat == 'Engine_Power_kW':
        val = st.number_input(feat, min_value =501.0252196 , max_value =2998.734329 ,value =1757.610939 , step =100.0 , help="انتاج الطاقة من المحرك الرئيسي اثناء الرحلة بالكيلووات")
    elif feat == 'Distance_Traveled_nm':
        val = st.number_input(feat, min_value =50.43314997, max_value =1998.337057 ,value =1036.406203 , step =10.0 , help="اجمالي الاميال البحرية التي سيتم اجتيازهااثناءالرحلة")
    elif feat == 'Draft_meters':
        val = st.number_input(feat, min_value =5.001946569, max_value =14.99294749 ,value =9.929102683 , step =1.0 , help="العمق الرأسي لهيكل السفينة تحت الماء")
    elif feat == 'Revenue_per_Voyage_USD':
        val = st.number_input(feat, min_value =50351.81445 , max_value =999916.6961 ,value =521362.062 , step =100.0 , help="الدخل المتولد من الرحلة(عقود الشحن)")
    elif feat == 'Turnaround_Time_hours':
        val = st.number_input(feat, min_value =12.01990927 , max_value =71.9724153 ,value =41.7475358 , step =1.0 , help="الوقت التس تقضيه في الميناء للشحن/التفريغ والتزود بالوقود")
    elif feat == 'Seasonal_Impact_Score':
        val = st.number_input(feat, min_value =1.003816044 , max_value =1.499223608 ,value =1.003816044 , step =0.01 , help="مقدار الموسمية التي اثرت على العمليات درجة مشتقة من 1-9")
    elif feat == 'Weekly_Voyage_Count':
        val = st.number_input(feat, min_value =1.0 , max_value =9.0 ,value =4.914839181 , step =0.1 , help="عدد الرحلات التي تكملها السفينة عادة في الاسبوع")
    elif feat == 'Average_Load_Percentage':
        val = st.number_input(feat, min_value =50.01200505 , max_value =99.99964331 ,value =75.21922177 , step =1.0 , help="مدى امتلاء السفينة كنسبة مئوية من السعة الكلية")
    else:
        val = st.number_input(feat, value=100.0, step=1.0)
    input_data[feat] = val

with col2:
# Categorical inputs
 st.subheader("🔤 Categorical Indicators")
 for feat, options in categorical_features.items():
    choice = st.selectbox(feat, options)
    input_data[feat] = choice

with col3:
# Prediction button
 if st.button("🔍 Predict Operational Group"):
    try:
        # Create input DataFrame
        df_input = pd.DataFrame([input_data])
        
        # One-hot encode (same as training)
        df_encoded = pd.get_dummies(df_input, columns=list(categorical_features.keys()))
        
        # Align columns: add missing one-hot columns as 0
        for col in all_onehot_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder to match training data
        df_encoded = df_encoded[all_onehot_features]
        
        # Scale
        X_scaled = scaler.transform(df_encoded)
        
        # Predict
        cluster_id = kmeans.predict(X_scaled)[0]
        cluster_label = cluster_labels[cluster_id]
        description = cluster_descriptions[cluster_label]
        
        # Display result
        st.success(f"✅ **Operational Group**: {cluster_label}")
        st.info(description)
        
        # Simulating Recommendations:
        if cluster_label == "High-Cost Carriers":
             st.warning("💡 **Recommendation**: This vessel shows signs of high operational cost. "
               "Consider engine retrofit or preventive maintenance.")
        elif cluster_label == "Cost-Efficient Carriers":
             st.success("💡 **Recommendation**: This is a benchmark vessel. "
               "Use its settings (e.g., load %, speed) as a standard for similar ships.")
        elif cluster_label == "Specialized Vessels":
             st.info("💡 **Recommendation**: This vessel is optimized for niche operations. "
            "Avoid assigning it to standard cargo routes.")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.code(str(e))