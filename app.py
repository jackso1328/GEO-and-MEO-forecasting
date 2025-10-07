# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load models
@st.cache_resource
def load_models():
    models = {}
    signals = ['x_error', 'y_error', 'z_error', 'satclockerror']
    for orbit in ['meo', 'geo']:
        models[orbit] = {}
        for s in signals:
            models[orbit][s] = joblib.load(f'{orbit}_{s}_model.pkl')
    return models

models = load_models()

st.title("🛰️ GNSS Error Predictor")
st.markdown("Predict satellite clock & ephemeris errors for MEO/GEO orbits")

# Inputs
orbit = st.selectbox("Satellite Orbit Type", ["MEO", "GEO"])
horizon = st.selectbox("Forecast Horizon", 
                       ["15min", "30min", "1h", "2h", "6h", "12h", "24h"])

# Map horizon to number of 15-min steps
horizon_map = {"15min":1, "30min":2, "1h":4, "2h":8, "6h":24, "12h":48, "24h":96}
steps = horizon_map[horizon]

# Assume prediction starts from Sept 8, 00:00
start_time = pd.Timestamp('2025-09-08 00:00')
times = pd.date_range(start=start_time, periods=steps, freq='15T')

# Get reference time (from training data start)
# You’ll need to store t0 during training — for now, hardcode approximate
t0 = pd.Timestamp('2025-09-01 14:00')  # from your data
future_seconds = ((times - t0).total_seconds()).values.reshape(-1, 1)

# Predict
orbit_key = orbit.lower()
results = {'Time': times}
for s in ['x_error', 'y_error', 'z_error', 'satclockerror']:
    pred = models[orbit_key][s].predict(future_seconds)
    results[s] = pred

df_pred = pd.DataFrame(results)

# Display
st.subheader(f"Predicted Errors ({horizon} ahead)")
st.dataframe(df_pred.style.format("{:.6f}"))

# Plot
st.subheader("Error Trends")
st.line_chart(df_pred.set_index('Time')[['x_error', 'y_error', 'z_error']])
st.line_chart(df_pred.set_index('Time')['satclockerror'])