import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('path_to_your_file/forestfires.csv')

# Preprocess the data as done before
label_encoder = LabelEncoder()
df['month'] = label_encoder.fit_transform(df['month'])
df['day'] = label_encoder.fit_transform(df['day'])

# Split the data into features and target
X = df.drop(columns=['area'])
y = df['area']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Streamlit UI
st.title("Forest Fire Prediction Dashboard")

st.sidebar.header("Input Parameters")

# Collect user input for prediction
X_input = {
    'X': st.sidebar.slider('X Coordinate', min_value=0, max_value=100, value=7),
    'Y': st.sidebar.slider('Y Coordinate', min_value=0, max_value=100, value=5),
    'FFMC': st.sidebar.slider('FFMC', min_value=50, max_value=100, value=86.2),
    'DMC': st.sidebar.slider('DMC', min_value=0, max_value=100, value=26.2),
    'DC': st.sidebar.slider('DC', min_value=0, max_value=100, value=94.3),
    'ISI': st.sidebar.slider('ISI', min_value=0.0, max_value=20.0, value=5.1),
    'temp': st.sidebar.slider('Temperature (°C)', min_value=0.0, max_value=40.0, value=8.2),
    'RH': st.sidebar.slider('Relative Humidity (%)', min_value=0, max_value=100, value=51),
    'wind': st.sidebar.slider('Wind Speed (km/h)', min_value=0.0, max_value=30.0, value=6.7),
    'rain': st.sidebar.slider('Rain (mm)', min_value=0.0, max_value=10.0, value=0.0),
}

# Encoding month and day features based on user input
month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_input = [0] * 12
for i, month in enumerate(month_list):
    if st.sidebar.checkbox(f'{month.capitalize()}'):
        month_input[i] = 1

day_list = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
day_input = [0] * 7
for i, day in enumerate(day_list):
    if st.sidebar.checkbox(f'{day.capitalize()}'):
        day_input[i] = 1

# Add month and day inputs to the input data
X_input.update(dict(zip([f'month_{month}' for month in month_list], month_input)))
X_input.update(dict(zip([f'day_{day}' for day in day_list], day_input)))

# Create the input DataFrame
input_df = pd.DataFrame([X_input])

# Make prediction
risk_pred = clf.predict(input_df)

# Display the predicted fire risk level
st.write(f"Predicted Fire Risk Level: {risk_pred[0]}")
# Set up default values for one-hot encoded months/days
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

for month in months:
    X_input[f'month_{month}'] = 0
for day in days:
    X_input[f'day_{day}'] = 0

# Set selected values to 1
if st.sidebar.checkbox('January'):
    X_input['month_jan'] = 1
# Repeat for other months & days...

# Create DataFrame
input_df = pd.DataFrame([X_input])

# Make prediction
try:
    risk_pred = clf.predict(input_df)
    st.success(f"🔥 Predicted Fire Risk Level: {risk_pred[0]}")
except Exception as e:
    st.error(f"Error during prediction: {e}")
