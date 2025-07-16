# 🚗 Driver Safety System Using PPG Signals

80% of daily accidents are caused due to one of the following: 

**-- Stress
--Drowsiness
--Burst Activity (Burst activity increases in case the driver is under the influence)**

A real-time driver monitoring system that uses PPG (Photoplethysmography) signals to detect stress, drowsiness, and burst activity. The system uses machine learning models (Random Forest and MLP), an interactive Streamlit UI, and Twilio API integration to send alerts if prolonged stress or fatigue is detected.

**🧠 Features**

- 🔍 **Stress Detection** using Random Forest
- 😴 **Drowsiness & Burst Activity Detection** using MLP
- 📈 **Live Monitoring Interface** built with Streamlit
- ⏱️ **ETA-Based Monitoring** from trip start to destination (calculates current location to destination time ETA in middle of the journey)
- 📲 **Twilio SMS Alerts** to guardian if stress persists >1 hour
- 🔁 **Checks every 5 minutes** throughout the journey


Happy Travelling :D

