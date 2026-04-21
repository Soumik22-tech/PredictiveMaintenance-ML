import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

@st.cache_resource
def load_ml_components():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_ml_components()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure models are trained and saved in the 'models/' folder.")
    st.stop()

st.title("⚙️ Predictive Maintenance of Industrial Machinery")
st.markdown("### Enter sensor readings to predict machine failure type")

st.sidebar.title("📊 Model Info")
st.sidebar.markdown("""
**Algorithm:** Random Forest  
**Accuracy:** 94.3%  
**F1-Macro:** 0.664

**Failure Classes:**
- No Failure
- Heat Dissipation Failure
- Overstrain Failure
- Power Failure
- Tool Wear Failure
- Random Failures
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Process Parameters")
    machine_type = st.selectbox("Machine Type", options=["L", "M", "H"])
    air_temp = st.slider("Air Temperature (K)", min_value=295.0, max_value=305.0, value=300.0, step=0.1)
    process_temp = st.slider("Process Temperature (K)", min_value=305.0, max_value=315.0, value=310.0, step=0.1)

with col2:
    st.subheader("Mechanical Parameters")
    rot_speed = st.slider("Rotational Speed (rpm)", min_value=1168, max_value=2886, value=1500, step=1)
    torque = st.slider("Torque (Nm)", min_value=3.8, max_value=76.6, value=40.0, step=0.1)
    tool_wear = st.slider("Tool Wear (min)", min_value=0, max_value=253, value=100, step=1)

temp_diff = process_temp - air_temp
power = torque * rot_speed
wear_torque = tool_wear * torque

st.info(f"""
**Engineered Features (Calculated Automatically):**  
- **Temperature Difference:** {temp_diff:.2f} K  
- **Estimated Power:** {power:.2f} W  
- **Wear-Torque Factor:** {wear_torque:.2f}
""")

if st.button("🔍 Predict Failure Type"):
    type_map = {"H": 0, "L": 1, "M": 2}
    type_enc = type_map[machine_type]
    
    input_data = np.array([[
        type_enc,
        air_temp,
        process_temp,
        rot_speed,
        torque,
        tool_wear,
        temp_diff,
        power,
        wear_torque
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction_enc = model.predict(input_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_enc])[0]
    probs = model.predict_proba(input_scaled)[0]
    
    st.subheader("Prediction Results")
    
    if prediction_label == "No Failure":
        st.success(f"✅ **Prediction: {prediction_label}**")
        recommendation = "Machine is operating normally. Continue monitoring."
    else:
        st.error(f"❌ **Prediction: {prediction_label}**")
        
        rec_map = {
            "Heat Dissipation Failure": "Check cooling systems immediately.",
            "Overstrain Failure": "Reduce machine load. Inspect mechanical parts.",
            "Power Failure": "Inspect power supply and electrical connections.",
            "Tool Wear Failure": "Replace cutting tool immediately.",
            "Random Failures": "Run full diagnostic check on the machine."
        }
        recommendation = rec_map.get(prediction_label, "Investigate error code and perform maintenance.")

    st.markdown(f"**Recommendation:** {recommendation}")
    
    st.markdown("#### Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Failure Type': label_encoder.classes_,
        'Probability': probs
    }).set_index('Failure Type')
    st.bar_chart(prob_df)

st.divider()
st.caption("Built with Python & Scikit-learn | Dataset: Kaggle Machine Predictive Maintenance")


