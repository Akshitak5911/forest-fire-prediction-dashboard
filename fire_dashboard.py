import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv(r'C:\Users\akaks\OneDrive\Downloads\archive\forestfires.csv')

# Preprocess the data
label_encoder = LabelEncoder()
df['month'] = label_encoder.fit_transform(df['month'])
df['day'] = label_encoder.fit_transform(df['day'])

# Split the data into features and target
X = df.drop(columns=['area'])
y = df['area']

# Train the model
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X, y)

# Streamlit UI
st.title("🌲 Forest Fire Burned Area Prediction Dashboard")

st.sidebar.header("Input Parameters")

# Collect user input for prediction
X_input = {
    'X': st.sidebar.slider('X Coordinate', min_value=1, max_value=9, value=7),
    'Y': st.sidebar.slider('Y Coordinate', min_value=2, max_value=9, value=5),
    'month': st.sidebar.slider('Month (0 = Jan, 11 = Dec)', min_value=0, max_value=11, value=7),
    'day': st.sidebar.slider('Day (0 = Mon, 6 = Sun)', min_value=0, max_value=6, value=2),
    'FFMC': st.sidebar.slider('FFMC', min_value=18.7, max_value=96.20, value=86.2),
    'DMC': st.sidebar.slider('DMC', min_value=1.1, max_value=291.3, value=26.2),
    'DC': st.sidebar.slider('DC', min_value=7.9, max_value=860.6, value=94.3),
    'ISI': st.sidebar.slider('ISI', min_value=0.0, max_value=56.1, value=5.1),
    'temp': st.sidebar.slider('Temperature (°C)', min_value=2.2, max_value=33.3, value=8.2),
    'RH': st.sidebar.slider('Relative Humidity (%)', min_value=15, max_value=100, value=51),
    'wind': st.sidebar.slider('Wind Speed (km/h)', min_value=0.4, max_value=9.4, value=6.7),
    'rain': st.sidebar.slider('Rain (mm)', min_value=0.0, max_value=6.4, value=0.0),
}

# Convert input into DataFrame
input_df = pd.DataFrame([X_input])

# Make prediction
predicted_area = clf.predict(input_df)[0]

# Display result
st.subheader("🔥 Predicted Burned Area:")
st.write(f"Estimated area burned: **{predicted_area:.2f} hectares**")
