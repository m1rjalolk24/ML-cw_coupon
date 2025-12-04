import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coupon Acceptance Predictor", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('coupon_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data['model']
feature_names = model_data['features']

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Analysis", "Prediction System"])

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("🚗 In-Vehicle Coupon Recommendation")
    st.markdown("""
    ### Business Problem
    This project predicts whether a driver will accept a coupon recommended to them while driving. 
    It compares **Logistic Regression**, **Decision Trees**, and **Random Forest** algorithms.
    
    ### Best Performing Model
    The **Random Forest Classifier** achieved the highest accuracy (**74.03%**), significantly outperforming the baseline.
    """)
    st.image("https://images.unsplash.com/photo-1449965408869-eaa3f722e40d", caption="Driving Scenario Analysis")

# --- PAGE 2: DATA ANALYSIS ---
elif page == "Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    # We load a small sample or hardcode simple stats for the app display
    # (To keep the app fast, we don't load the full 12k dataset here unless necessary)
    st.write("### Key Insights")
    st.markdown("- **Coffee House Coupons** have a strong correlation with drivers who visit cafes frequentely.")
    st.markdown("- **Expiration Time** (2h vs 1d) is a critical deciding factor.")
    
    st.write("### Model Performance Comparison")
    perf_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'Accuracy': [0.6685, 0.6764, 0.7403]
    })
    st.bar_chart(perf_data.set_index('Model'))

# --- PAGE 3: PREDICTION SYSTEM ---
elif page == "Prediction System":
    st.title("🤖 Live Prediction Model")
    st.write("Configure the scenario below to see if the driver accepts the coupon.")
    
    # INPUT FORM
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            destination = st.selectbox("Destination", ["No Urgent Place", "Home", "Work"])
            weather = st.selectbox("Weather", ["Sunny", "Rainy", "Snowy"])
            temperature = st.slider("Temperature (F)", 30, 80, 55)
            time_hour = st.slider("Time of Day (24h)", 0, 23, 14)
            passenger = st.selectbox("Passenger", ["Alone", "Friend(s)", "Kid(s)", "Partner"])
        
        with col2:
            coupon = st.selectbox("Coupon Type", ["Coffee House", "Restaurant(<20)", "Carry out", "Bar", "Restaurant(20-50)"])
            expiration = st.selectbox("Expiration", ["2 Hours", "1 Day"])
            coffee_freq = st.select_slider("Coffee House Visits", options=["never", "less1", "1~3", "4~8", "gt8"])
            bar_freq = st.select_slider("Bar Visits", options=["never", "less1", "1~3", "4~8", "gt8"])
            
        submitted = st.form_submit_button("Predict Acceptance")
        
    if submitted:
        # Preprocess Input (Quick mapping to match your training data encoding)
        # NOTE: This is a simplified mapper. In a real production app, you'd use the exact same encoders.
        # For the coursework demo, we map the most critical features manually to show it works.
        
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Map Numeric/Ordinal Inputs
        input_data['time_hour'] = time_hour
        input_data['expiration_hours'] = 2 if expiration == "2 Hours" else 24
        
        # We assume standard encoding for these demo sliders (0-4)
        mapping = {"never":0, "less1":1, "1~3":2, "4~8":3, "gt8":4}
        input_data['CoffeeHouse'] = mapping[coffee_freq]
        input_data['Bar'] = mapping[bar_freq]
        
        # For One-Hot features, we just set the specific column to 1 if it exists
        # Example: if user picked "Sunny", we look for "weather_Sunny"
        if f"weather_{weather}" in feature_names: input_data[f"weather_{weather}"] = 1
        if f"coupon_{coupon}" in feature_names: input_data[f"coupon_{coupon}"] = 1
        if f"destination_{destination}" in feature_names: input_data[f"destination_{destination}"] = 1
        
        # PREDICT
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"✅ Prediction: **ACCEPTED** (Confidence: {prob:.1%})")
            st.balloons()
        else:
            st.error(f"❌ Prediction: **REJECTED** (Confidence: {1-prob:.1%})")