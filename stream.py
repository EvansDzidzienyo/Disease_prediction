"""Streamlit interface for the reproducibly trained disease-presence model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "classification_data.csv"
MODEL_PATH = ROOT / "artifacts" / "disease_presence_pipeline.joblib"

st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱", layout="wide")
st.title("Crop Disease Presence — Experimental Model")
st.caption("A machine-learning demonstration using the project's agricultural dataset.")
st.warning(
    "This prototype is not an agronomic diagnosis. Verify field conditions with "
    "qualified agricultural extension support before taking action."
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    data = load_data()
    model = load_model()
except Exception:
    st.error(
        "The dataset or trained pipeline could not be loaded. Run python train.py "
        "and check that the required repository files are present."
    )
    st.stop()

required_columns = [
    "Region", "Crop_Type", "Fertilizer_Used", "Pest_Presence",
    "Date", "Rainfall_mm", "Temperature_C", "Soil_Moisture",
]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error("The dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")
if data["Date"].isna().any():
    st.error("The dataset contains dates outside the expected MM/DD/YYYY format.")
    st.stop()
data["Day"] = data["Date"].dt.day

st.subheader("Dataset summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average rainfall", f"{data['Rainfall_mm'].mean():.2f} mm")
with col2:
    st.metric("Average temperature", f"{data['Temperature_C'].mean():.2f} °C")
with col3:
    st.metric("Average soil moisture", f"{data['Soil_Moisture'].mean():.2f}")

st.sidebar.header("Model input features")
region = st.sidebar.selectbox("Region", sorted(data["Region"].dropna().unique()))
crop_type = st.sidebar.selectbox("Crop type", sorted(data["Crop_Type"].dropna().unique()))
fertilizer_use = st.sidebar.selectbox(
    "Fertilizer used", sorted(data["Fertilizer_Used"].dropna().unique())
)
pest_presence = st.sidebar.selectbox(
    "Pest presence", sorted(data["Pest_Presence"].dropna().unique())
)
day = st.sidebar.slider("Day of month", 1, 31, 15)
rainfall = st.sidebar.slider(
    "Rainfall (mm)", float(data["Rainfall_mm"].min()),
    float(data["Rainfall_mm"].max()), float(data["Rainfall_mm"].median())
)
temperature = st.sidebar.slider(
    "Temperature (°C)", float(data["Temperature_C"].min()),
    float(data["Temperature_C"].max()), float(data["Temperature_C"].median())
)
soil_moisture = st.sidebar.slider(
    "Soil moisture", float(data["Soil_Moisture"].min()),
    float(data["Soil_Moisture"].max()), float(data["Soil_Moisture"].median())
)

user_input = pd.DataFrame([{
    "Region": region,
    "Crop_Type": crop_type,
    "Fertilizer_Used": fertilizer_use,
    "Pest_Presence": pest_presence,
    "Day": day,
    "Rainfall_mm": rainfall,
    "Temperature_C": temperature,
    "Soil_Moisture": soil_moisture,
}])

if st.button("Predict disease presence"):
    try:
        expected = list(model.feature_names_in_)
        if set(expected) != set(user_input.columns):
            st.error("App inputs do not match the trained model's feature schema.")
            st.stop()
        prediction = int(model.predict(user_input[expected])[0])
        st.subheader("Model output")
        st.write(f"Predicted Disease_Presence class: **{prediction}**")
        st.caption(
            "This is the dataset's raw 0/1 target class. The repository does not "
            "provide verified semantic documentation for interpreting the class."
        )
    except Exception:
        st.error("Prediction failed. Check the saved model and app input compatibility.")
