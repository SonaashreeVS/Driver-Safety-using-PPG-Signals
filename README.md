# 🚗 Driver Safety System Using PPG Signals (Refer Below for Dataset)

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

**Dataset:**
https://www.kaggle.com/datasets/sonsas/photoplethysmography-signals-mental-workload

The above dataset contains 24 people's (22 Men and 2 Women) ppg signals taken during high mental workload and low mental workload neatly arranged into approximately **76800 rows** that has helped me to achieve **100% accuracy** in my stress calculator and gave me the best accuracy possible for my drowsiness and burst activity detector model. 
Happy Travelling :D

