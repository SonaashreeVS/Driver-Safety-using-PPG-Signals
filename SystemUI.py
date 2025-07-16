import streamlit as st
import pandas as pd
import numpy as np
from PyEMD import EMD
from scipy.stats import skew
import joblib
from datetime import datetime, timedelta
from twilio.rest import Client
from haversine import haversine
import requests
import time
import os
import streamlit.components.v1 as components
import folium
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE = ""
stress_model = joblib.load(r"C:\\Users\\sonaa\\clean_mwl_classifier.joblib")
drowsy_model = joblib.load(r"C:\\Users\\sonaa\\drowsy_burst_mlp_model.joblib")
eta_model = joblib.load(r"C:\\Users\\sonaa\\eta_lightgbm_model.pkl")
#indian numbers formatting
def format_number(number):
    number = number.strip()
    if not number.startswith('+'):
        number = '+91' + number
    return number
def send_sms(to, message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(to=to, from_=TWILIO_PHONE, body=message)
        st.success(f"✅ SMS sent to {to}")
        return True
    except Exception as e:
        st.error(f"❌ SMS error: {e}")
        return False
def classify_ppg_segment(ppg_segment):
    emd = EMD(spline_kind='cubic', MAX_ITERATION=100)
    imfs = emd(ppg_segment, max_imf=10)
    if imfs.shape[0] < 2:
        return None
    imf1 = imfs[0]
    features = np.array([
        np.mean(imf1),
        np.min(imf1),
        np.max(imf1),
        skew(imf1)
    ]).reshape(1, -1)
    stress_pred = stress_model.predict(features)[0]
    stress_prob = stress_model.predict_proba(features)[0]
    drowsy_pred = drowsy_model.predict(features)[0]
    drowsy_prob = drowsy_model.predict_proba(features)[0]
    if stress_pred == 0 and drowsy_pred == 1:
        stress_pred = 1
        stress_prob = [0.0, 1.0]
    return stress_pred, stress_prob, drowsy_pred, drowsy_prob
def geocode_photon(address):
    bbox = "79.00,13.05,79.30,12.85"
    url = f"https://photon.komoot.io/api/?q={address}&limit=1&bbox={bbox}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["features"]:
            coords = data["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]
    except:
        return None
    return None
def is_within_vellore(lat, lng):
    return 12.85 <= lat <= 13.05 and 79.00 <= lng <= 79.30
def get_congestion_label():
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    if day < 5 and (8 <= hour <= 11 or 17 <= hour <= 20):
        return "🔴 High Congestion"
    elif day < 5 and (7 <= hour < 8 or 15 <= hour < 17):
        return "🟠 Moderate Congestion"
    elif day >= 5 and (10 <= hour <= 13 or 18 <= hour <= 21):
        return "🟡 Weekend Busy"
    else:
        return "🟢 Low Congestion"
def predict_eta(start_lat, start_lng, end_lat, end_lng):
    h_km = haversine((start_lat, start_lng), (end_lat, end_lng))
    data = pd.DataFrame([{
        'start_lat': start_lat,
        'start_lng': start_lng,
        'end_lat': end_lat,
        'end_lng': end_lng,
        'haversine_km': h_km
    }])
    eta_seconds = eta_model.predict(data)[0]
    return round(eta_seconds / 60, 2), round(h_km, 2)  # minutes, km
# UI
st.set_page_config(page_title="Driver Safety", layout="centered")
st.title("Driver Safety Alert System")
if "stress_log" not in st.session_state:
    st.session_state.stress_log = []
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
# sign-in
with st.form("combined_form"):
    st.subheader("Driver & Trip Details")
    name = st.text_input("Driver Name")
    phone = st.text_input("Driver Phone")
    guardian = st.text_input("Guardian Phone")
    start_address = st.text_input("Start Location", "VIT Vellore")
    end_address = st.text_input("Destination", "Katpadi Junction")
    file = st.file_uploader("Upload 768-sample PPG CSV", type="csv")
    submitted = st.form_submit_button("Analyze Once")
#start and stop
col1, col2 = st.columns(2)
if col1.button("▶️ Start Monitoring"):
    st.session_state.monitoring = True
    st.success("Monitoring started.")
if col2.button("⏹️ Stop Monitoring"):
    st.session_state.monitoring = False
    st.success("Monitoring stopped.")
if submitted or st.session_state.monitoring:
    if not name or not phone or not guardian or not file:
        st.warning("Please complete all fields.")
    else:
        try:
            df = pd.read_csv(file)
            ppg = df.iloc[:768, 0].dropna().values
            if len(ppg) != 768:
                st.error("CSV must contain exactly 768 samples.")
            else:
                result = classify_ppg_segment(ppg)
                if result:
                    stress, stress_conf, drowsy, drowsy_conf = result
                    stress_percent = round(stress_conf[stress] * 100, 2)
                    drowsy_label = "Burst Activity" if drowsy else "Drowsy"
                    st.success(f"**Stress Level:** {'High' if stress else 'Low'} ({stress_percent}%)")
                    st.info(f"**Driver State:** {drowsy_label}")
                    current_coords = geocode_photon(start_address)
                    end_coords = geocode_photon(end_address)
                    if current_coords and end_coords:
                        s_lat, s_lng = current_coords
                        e_lat, e_lng = end_coords
                        if not (is_within_vellore(s_lat, s_lng) and is_within_vellore(e_lat, e_lng)):
                            st.error("Destination is out of supported region.")
                        else:
                            eta_min, dist_km = predict_eta(s_lat, s_lng, e_lat, e_lng)
                            congestion = get_congestion_label()
                            st.markdown(f"Estimated Arrival Time: **{eta_min} min**")
                            st.markdown(f"Distance: **{dist_km} km**")
                            st.markdown(f"🚦 Traffic: {congestion}")
                            phone = format_number(phone)
                            guardian = format_number(guardian)
                            gmap_link = f"https://www.google.com/maps?q={s_lat},{s_lng}"
                            if stress == 1 or drowsy == 1:
                                st.session_state.stress_log.append(datetime.now())
                                msg = f"You're currently under {stress_percent}% stress. Please consider taking a break from driving.\n📍 Location: {gmap_link}"
                                send_sms(phone, msg)
                            if st.session_state.stress_log:
                                window = [t for t in st.session_state.stress_log if datetime.now() - t <= timedelta(hours=1)]
                                if len(window) >= 12:
                                    guardian_msg = (
                                        f"⚠️ {name} is under high stress ({stress_percent}%) for 1+ hour.\n"
                                        f"📍 Last known location: {gmap_link}\n"
                                        f"🕒 They will arrive in approximately {eta_min} minutes.\n"
                                        f"Please consider checking up on them."
                                    )
                                    send_sms(guardian, guardian_msg)
                                    st.session_state.stress_log.clear()
                            if st.session_state.monitoring:
                                with st.spinner("⏳ Sleeping for 5 minutes..."):
                                    time.sleep(300)
                                st.experimental_rerun()
                else:
                    st.error("Could not extract valid features.")
        except Exception as e:
            st.error(f"Error: {e}")
