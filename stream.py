import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱", layout="wide")
st.title("Crop Disease Presence — Experimental Model")
st.caption("A machine-learning demonstration using the project's agricultural dataset.")
st.warning("This prototype is not an agronomic diagnosis. Verify field conditions with qualified agricultural extension support before taking action.")

@st.cache_data
def load_data():
    return pd.read_csv("classification_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("pest_model.joblib")

try:
    data = load_data()
    model = load_model()
except Exception as exc:
    st.error("The dataset or saved model could not be loaded. Check that the repository files and compatible package versions are available.")
    st.exception(exc)
    st.stop()

required_columns = [
    "Region", "Crop_Type", "Fertilizer_Used", "Pest_Presence",
    "Day", "Rainfall_mm", "Temperature_C", "Soil_Moisture",
]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error("The dataset is missing required model input columns: " + ", ".join(missing_columns))
    st.stop()

st.subheader("Dataset summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average rainfall", f"{data['Rainfall_mm'].mean():.2f} mm")
with col2:
    st.metric("Average temperature", f"{data['Temperature_C'].mean():.2f} °C")
with col3:
    st.metric("Average soil moisture", f"{data['Soil_Moisture'].mean():.2f} %")

st.sidebar.header("Model input features")
region = st.sidebar.selectbox("Region", sorted(data["Region"].dropna().unique()))
crop_type = st.sidebar.selectbox("Crop type", sorted(data["Crop_Type"].dropna().unique()))
fertilizer_use = st.sidebar.selectbox("Fertilizer used", sorted(data["Fertilizer_Used"].dropna().unique()))
pest_presence = st.sidebar.selectbox("Pest presence", sorted(data["Pest_Presence"].dropna().unique()))
day = st.sidebar.slider("Day of month", 1, 31, 15)
rainfall = st.sidebar.slider("Rainfall (mm)", float(data["Rainfall_mm"].min()), float(data["Rainfall_mm"].max()), float(data["Rainfall_mm"].mean()))
temperature = st.sidebar.slider("Temperature (°C)", float(data["Temperature_C"].min()), float(data["Temperature_C"].max()), float(data["Temperature_C"].mean()))
soil_moisture = st.sidebar.slider("Soil moisture (%)", float(data["Soil_Moisture"].min()), float(data["Soil_Moisture"].max()), float(data["Soil_Moisture"].mean()))

# The training notebook defines Disease_Presence as y. Pest_Presence is an input feature.
user_input = pd.DataFrame({
    "Region": [region],
    "Crop_Type": [crop_type],
    "Fertilizer_Used": [fertilizer_use],
    "Pest_Presence": [pest_presence],
    "Day": [day],
    "Rainfall_mm": [rainfall],
    "Temperature_C": [temperature],
    "Soil_Moisture": [soil_moisture],
})

if st.button("Predict disease presence"):
    try:
        prediction = model.predict(user_input)
        label = prediction[0]
        st.subheader("Model output")
        st.write(f"Predicted Disease_Presence class: **{label}**")
        st.caption("The training notebook uses Disease_Presence as the target. Confirm the dataset's class encoding before interpreting 0/1 as absence/presence.")
    except Exception as exc:
        st.error("Prediction failed. The app inputs may not match the preprocessing pipeline stored in pest_model.joblib.")
        st.exception(exc)
