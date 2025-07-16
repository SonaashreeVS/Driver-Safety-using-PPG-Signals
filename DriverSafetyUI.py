# Streamlit Driver Monitoring App with Cross Verification
import streamlit as st
import pandas as pd
import numpy as np
from PyEMD import EMD
from scipy.stats import skew
import joblib
from datetime import datetime, timedelta
from twilio.rest import Client

# ------------------- TWILIO CONFIG --------------------
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE = ""

# ------------------- LOAD MODELS --------------------
stress_model = joblib.load(r"C:\\Users\\sonaa\\clean_mwl_classifier.joblib")
drowsy_model = joblib.load(r"C:\\Users\\sonaa\\drowsy_burst_mlp_model.joblib")

# ------------------- SMS FUNCTION --------------------
stress_timestamps = []
def send_sms(to, message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(to=to, from_=TWILIO_PHONE, body=message)
        return True
    except Exception as e:
        st.error(f"SMS sending failed: {e}")
        return False

# ------------------- CLASSIFICATION --------------------
def classify_ppg_segment(ppg_segment):
    emd = EMD(spline_kind='cubic', MAX_ITERATION=100)
    imfs = emd(ppg_segment, max_imf=10)

    if imfs.shape[0] < 2:
        return None

    imf1 = imfs[0]

    # Shared feature set
    features = np.array([
        np.mean(imf1),
        np.min(imf1),
        np.max(imf1),
        skew(imf1)
    ]).reshape(1, -1)

    # Predictions
    stress_pred = stress_model.predict(features)[0]
    stress_prob = stress_model.predict_proba(features)[0]

    drowsy_pred = drowsy_model.predict(features)[0]
    drowsy_prob = drowsy_model.predict_proba(features)[0]

    # Cross verification logic: if burst is detected, override stress
    if stress_pred == 0 and drowsy_pred == 1:
        stress_pred = 1
        stress_prob = [0.0, 1.0]

    return stress_pred, stress_prob, drowsy_pred, drowsy_prob

# ------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Driver Monitoring", layout="centered")
st.title("Driver Safety & Fatigue Detection")

with st.form("driver_form"):
    st.subheader("Driver Details")
    name = st.text_input("Driver Name")
    phone = st.text_input("Driver Phone Number")
    guardian = st.text_input("Guardian Phone Number (for Emergency alerts)")
    file = st.file_uploader("Upload 768-sample PPG CSV", type="csv")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not name or not phone or not file:
        st.warning("Please complete all fields and upload a CSV.")
    else:
        try:
            df = pd.read_csv(file)
            ppg = df.iloc[:768, 0].dropna().values

            if len(ppg) != 768:
                st.error("CSV must contain exactly 768 samples.")
            else:
                result = classify_ppg_segment(ppg)

                if result is None:
                    st.error("Could not extract valid features.")
                else:
                    stress, stress_conf, drowsy, drowsy_conf = result

                    st.success(f"Prediction for {name}:")
                    st.markdown(f"**Stress Level**: {'High' if stress else 'Low'} ({round(stress_conf[stress]*100, 2)}%)")
                    st.markdown(f"**Driver State**: {'Burst Activity' if drowsy else 'Drowsy'} ({round(drowsy_conf[drowsy]*100, 2)}%)")

                    if stress == 1 and stress_conf[1] > 0.95:
                        if send_sms(phone, f" Alert: {name} is under high stress while driving."):
                            st.info("Alert sent to driver's phone.")
                            stress_timestamps.append(datetime.now())

                    if stress_timestamps:
                        if datetime.now() - stress_timestamps[-1] > timedelta(hours=1):
                            if send_sms(guardian, f" Guardian Alert: {name} has been stressed for over 1 hour.Driver Safety should be monitored properly"):
                                st.warning(" Alert sent to guardian.")
                                stress_timestamps.clear()

        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption("Built with Streamlit, PyEMD, and ML for driver monitoring and safety purposes")
