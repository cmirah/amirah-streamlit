import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="SIRF Prediction", page_icon="📊")
st.title("Prediction of COVID-19 Disease using SIRF Model")

def train_model(df, features, target):
    # Feature engineering and preprocessing
    X = df[features]
    y = df[target]
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestRegressor()
    model.fit(X_scaled, y)
    return model, scaler

def predict_value(model, scaler, inputs):
    # Scale the input data
    new_data = np.array([inputs])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_value = model.predict(new_data_scaled)
    return predicted_value[0]

def main():
    # Input form for new data
    prediction_date = st.date_input('Prediction Date')

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

    # Display the DataFrame for debugging
    st.write("DataFrame head:")
    st.write(df.head())
    st.write("DataFrame tail:")
    st.write(df.tail())
    st.write("DataFrame date types:")
    st.write(df['date'].dtype)

    # Sort data by date
    df = df.sort_values('date')

    # Train models
    models = {}
    scalers = {}
    targets = ['susceptible', 'infected', 'recovered', 'fatal']
    features_map = {
        'susceptible': ['infected', 'recovered', 'fatal', 'confirmed'],
        'infected': ['susceptible', 'recovered', 'fatal', 'confirmed'],
        'recovered': ['susceptible', 'infected', 'fatal', 'confirmed'],
        'fatal': ['susceptible', 'infected', 'recovered', 'confirmed']
    }
    
    for target in targets:
        models[target], scalers[target] = train_model(df, features_map[target], target)

    # Ensure prediction_date is a datetime object
    prediction_date = pd.to_datetime(prediction_date)

    # Get the latest data before the prediction date
    latest_data = df[df['date'] < prediction_date]
    
    if latest_data.empty:
        st.warning("No data available before the selected prediction date. Please choose a different date.")
        return

    latest_data = latest_data.iloc[-1]
    
    # Predict values
    inputs_map = {
        'susceptible': [latest_data['infected'], latest_data['recovered'], latest_data['fatal'], latest_data['confirmed']],
        'infected': [latest_data['susceptible'], latest_data['recovered'], latest_data['fatal'], latest_data['confirmed']],
        'recovered': [latest_data['susceptible'], latest_data['infected'], latest_data['fatal'], latest_data['confirmed']],
        'fatal': [latest_data['susceptible'], latest_data['infected'], latest_data['recovered'], latest_data['confirmed']]
    }

    if st.button('Predict SIRF'):
        predictions = {}
        for target in targets:
            inputs = inputs_map[target]
            predictions[target] = predict_value(models[target], scalers[target], inputs)
        
        st.success(f"Predicted values on {prediction_date.date()} are:")
        st.write(f"Susceptible: {predictions['susceptible']:.0f}")
        st.write(f"Infected: {predictions['infected']:.0f}")
        st.write(f"Recovered: {predictions['recovered']:.0f}")
        st.write(f"Fatal: {predictions['fatal']:.0f}")

if __name__ == '__main__':
    main()

