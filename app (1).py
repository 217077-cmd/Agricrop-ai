
# ============================================================
# STREAMLIT WEB APP: CROP RECOMMENDATION AI PLATFORM
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

model_package = joblib.load("crop_recommendation_model.pkl")

model = model_package["model"]
features = model_package["features"]
classes = model_package["classes"]

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="AgriCrop AI",
    page_icon="🌱",
    layout="wide"
)

# ------------------------------------------------------------
# Custom title
# ------------------------------------------------------------

st.title("🌱 AgriCrop AI")
st.subheader("Smart Crop Recommendation System")

st.write(
    "This web application predicts the most suitable crop based on soil nutrients "
    "and climate conditions using a machine learning classification model."
)

st.divider()

# ------------------------------------------------------------
# Sidebar information
# ------------------------------------------------------------

st.sidebar.title("About")
st.sidebar.write(
    "This system uses a Random Forest Classifier to recommend crops based on "
    "N, P, K, temperature, humidity, pH, and rainfall."
)

st.sidebar.write("Developed for agriculture AI class project.")

# ------------------------------------------------------------
# Manual prediction
# ------------------------------------------------------------

st.header("Single Crop Prediction")

st.write("Enter the soil and climate values below.")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input(
        "Nitrogen (N)",
        min_value=0,
        max_value=200,
        value=50
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0
    )

with col2:
    P = st.number_input(
        "Phosphorus (P)",
        min_value=0,
        max_value=200,
        value=50
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with col3:
    K = st.number_input(
        "Potassium (K)",
        min_value=0,
        max_value=250,
        value=50
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

input_data = pd.DataFrame({
    "N": [N],
    "P": [P],
    "K": [K],
    "temperature": [temperature],
    "humidity": [humidity],
    "ph": [ph],
    "rainfall": [rainfall]
})

st.write("Input Data Preview:")
st.dataframe(input_data, use_container_width=True)

# ------------------------------------------------------------
# Prediction button
# ------------------------------------------------------------

if st.button("Predict Suitable Crop"):
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities.max() * 100

    st.success(f"Recommended Crop: **{prediction.upper()}**")
    st.info(f"Model Confidence: **{confidence:.2f}%**")

    probability_df = pd.DataFrame({
        "Crop": classes,
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    ).head(5)

    probability_df["Probability (%)"] = probability_df["Probability"] * 100

    st.subheader("Top 5 Crop Recommendations")
    st.dataframe(
        probability_df[["Crop", "Probability (%)"]],
        use_container_width=True
    )

    st.bar_chart(probability_df.set_index("Crop")["Probability (%)"])

st.divider()

# ------------------------------------------------------------
# Batch prediction
# ------------------------------------------------------------

st.header("Batch Prediction")

st.write(
    "Upload a CSV file containing these columns: "
    "`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(batch_df, use_container_width=True)

    missing_columns = [col for col in features if col not in batch_df.columns]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
    else:
        batch_predictions = model.predict(batch_df[features])
        batch_probabilities = model.predict_proba(batch_df[features]).max(axis=1) * 100

        batch_df["Recommended Crop"] = batch_predictions
        batch_df["Confidence (%)"] = batch_probabilities

        st.success("Batch prediction completed.")

        st.subheader("Prediction Results")
        st.dataframe(batch_df, use_container_width=True)

        csv = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="crop_recommendation_results.csv",
            mime="text/csv"
        )

st.divider()

# ------------------------------------------------------------
# About model
# ------------------------------------------------------------

st.header("About This Model")

st.write("""
This crop recommendation system uses a Random Forest Classifier.

Input variables:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Soil pH
- Rainfall

Output:

- Recommended crop
- Model confidence
- Top 5 crop suggestions

This system is a decision-support tool. It should not fully replace expert agronomic advice because real farm conditions may also depend on crop variety, pests, diseases, irrigation, soil texture, soil salinity, and market demand.
""")
