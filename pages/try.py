import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="SIRF Prediction", page_icon="📊")
st.title("Prediction of COVID-19 Disease using SIRF Model")

def train_model(df, target):
    # Feature engineering and preprocessing
    X = df[['susceptible','infected', 'recovered', 'fatal','confirmed']]
    y = df[target]
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestRegressor()
    model.fit(X_scaled, y)
    return model, scaler

def predict_value(model, scaler, infected, recovered, fatal):
    # Scale the input data
    new_data = np.array([[susceptible, infected, recovered, fatal, confirmed]])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_value = model.predict(new_data_scaled)
    return predicted_value[0]

def main():
    # Input form for new data
    susceptible = st.number_input('Susceptible', value=10000000)
    infected = st.number_input('Infected', value=1000)
    recovered = st.number_input('Recovered', value=1000)
    fatal = st.number_input('Fatal', value=50)
    confirmed = st.number_input('Confirmed', value=100000)
    prediction_date = st.date_input('Prediction Date')

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Train models
    models = {}
    scalers = {}
    targets = ['susceptible', 'infected', 'recovered', 'fatal']
    for target in targets:
        models[target], scalers[target] = train_model(df, target)

    # Predict button
    if st.button('Predict'):
        predictions = {}
        for target in targets:
            predictions[target] = predict_value(models[target], scalers[target], infected, recovered, fatal)

        # Display predictions
        st.success(f'Predicted Susceptible value on {prediction_date} is : {predictions["susceptible"]:.0f}')
        st.success(f'Predicted Infected value on {prediction_date} is : {predictions["infected"]:.0f}')
        st.success(f'Predicted Recovered value on {prediction_date} is : {predictions["recovered"]:.0f}')
        st.success(f'Predicted Fatal value on {prediction_date} is : {predictions["fatal"]:.0f}')

if __name__ == '__main__':
    main()






