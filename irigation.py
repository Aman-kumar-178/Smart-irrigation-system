import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# -------------------------------
# Load the trained model
# -------------------------------
model = joblib.load("Farm_Irrigation_System.pkl")

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("💧 Smart Sprinkler System")
st.subheader("Predict Sprinkler Status using Sensor Data")

# Sidebar for input mode
st.sidebar.header("⚙️ Settings")
input_mode = st.sidebar.radio("Select Input Mode:", ["Sliders", "Random Values", "Upload CSV"])

# -------------------------------
# 1. Collect Sensor Inputs
# -------------------------------
sensor_values = []

if input_mode == "Sliders":
    st.markdown("### Adjust Sensor Values (0–1)")
    for i in range(20):
        val = st.slider(f"Sensor {i+1}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        sensor_values.append(val)

elif input_mode == "Random Values":
    if st.button("🎲 Generate Random Values"):
        sensor_values = list(np.round(np.random.rand(20), 2))
        st.write("Generated Random Sensor Values:", sensor_values)

elif input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with 20 sensor values", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 20:
            sensor_values = df.iloc[0].tolist()
            st.write("Uploaded Sensor Values:", sensor_values)
        else:
            st.error("CSV must contain exactly 20 columns (one for each sensor).")

# -------------------------------
# 2. Prediction
# -------------------------------
if sensor_values and st.button("🔮 Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(input_array)

    # If multi-output, take first row
    if prediction.ndim > 1:
        prediction = prediction[0]

    # Show predictions
    st.markdown("### 🚰 Sprinkler Status Prediction:")
    for i, status in enumerate(prediction):
        color = "green" if status == 1 else "red"
        st.markdown(f"<span style='color:{color}'>🌱 Sprinkler {i+1} (Parcel {i+1}): {'ON ✅' if status==1 else 'OFF ❌'}</span>", unsafe_allow_html=True)

    # Option to save results
    if st.button("💾 Save Results as CSV"):
        df_result = pd.DataFrame({
            "Sensor": [f"Sensor {i+1}" for i in range(20)],
            "Value": sensor_values,
            "Sprinkler_Status": ["ON" if s == 1 else "OFF" for s in prediction]
        })
        df_result.to_csv("sprinkler_results.csv", index=False)
        st.success("Results saved as sprinkler_results.csv")

# -------------------------------
# 3. Feature Importance (if supported)
# -------------------------------
if hasattr(model, "feature_importances_"):
    st.subheader("📊 Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(range(len(importance)), importance, color="skyblue")
    ax.set_xlabel("Sensors")
    ax.set_ylabel("Importance")
    ax.set_title("Sensor Importance for Sprinkler Decisions")
    st.pyplot(fig)

# -------------------------------
# 4. Optional Model Performance
# -------------------------------
if st.sidebar.checkbox("Show Model Performance (needs test data)"):
    st.info("⚠️ Load your X_test, y_test here to evaluate model performance.")
    # Example placeholder (replace with real test data)
    # y_pred = model.predict(X_test)
    # report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    # st.json(report)
