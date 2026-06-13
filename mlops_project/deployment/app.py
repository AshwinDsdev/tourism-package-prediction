import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")
st.title("Wellness Tourism Package Purchase Predictor")
st.markdown("Enter the customer details below to predict if they will purchase the new Wellness Tourism Package.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="ashwindatasense/wellness-tourism-model", filename="model.joblib")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model from Hugging Face: {e}")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
    pref_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=2)
    pitch_sat = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=20000.0)

with col2:
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns Car?", ["Yes", "No"])

predict_btn = st.button("Predict Purchase Probability")

if predict_btn:
    # Convert inputs to dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'DurationOfPitch': [duration_of_pitch],
        'NumberOfPersonVisiting': [num_person],
        'NumberOfFollowups': [num_followups],
        'PreferredPropertyStar': [pref_star],
        'NumberOfTrips': [num_trips],
        'PitchSatisfactionScore': [pitch_sat],
        'NumberOfChildrenVisiting': [num_children],
        'MonthlyIncome': [monthly_income],
        'TypeofContact': [contact],
        'Occupation': [occupation],
        'Gender': [gender],
        'ProductPitched': [product],
        'MaritalStatus': [marital],
        'Designation': [designation],
        'CityTier': [city_tier],
        'Passport': [1 if passport == "Yes" else 0],
        'OwnCar': [1 if own_car == "Yes" else 0]
    })
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.success(f"**High Likelihood of Purchase!** (Probability: {proba:.2%})")
        st.balloons()
    else:
        st.warning(f"**Low Likelihood of Purchase.** (Probability: {proba:.2%})")
