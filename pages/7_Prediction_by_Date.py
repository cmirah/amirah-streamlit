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
    susceptible = st.number_input('Susceptible', value=10000000)
    infected = st.number_input('Infected', value=1000)
    recovered = st.number_input('Recovered', value=1000)
    fatal = st.number_input('Fatal', value=50)
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
    features_map = {
        'susceptible': ['infected', 'recovered', 'fatal', 'confirmed'],
        'infected': ['susceptible', 'recovered', 'fatal', 'confirmed'],
        'recovered': ['susceptible', 'infected', 'fatal', 'confirmed'],
        'fatal': ['susceptible', 'infected', 'recovered', 'confirmed']
    }
    
    for target in targets:
        models[target], scalers[target] = train_model(df, features_map[target], target)

    # Predict button for S
    if st.button('Predict S'):
        inputs = [infected, recovered, fatal, 0]  # 0 for confirmed (not used for S prediction)
        prediction = predict_value(models['susceptible'], scalers['susceptible'], inputs)
        st.success(f'Predicted Susceptible value on {prediction_date} is : {prediction:.0f}')

    # Predict button for I
    if st.button('Predict I'):
        inputs = [susceptible, recovered, fatal, 0]  # 0 for confirmed (not used for I prediction)
        prediction = predict_value(models['infected'], scalers['infected'], inputs)
        st.success(f'Predicted Infected value on {prediction_date} is : {prediction:.0f}')

    # Predict button for R
    if st.button('Predict R'):
        inputs = [susceptible, infected, fatal, 0]  # 0 for confirmed (not used for R prediction)
        prediction = predict_value(models['recovered'], scalers['recovered'], inputs)
        st.success(f'Predicted Recovered value on {prediction_date} is : {prediction:.0f}')

    # Predict button for F
    if st.button('Predict F'):
        inputs = [susceptible, infected, recovered, 0]  # 0 for confirmed (not used for F prediction)
        prediction = predict_value(models['fatal'], scalers['fatal'], inputs)
        st.success(f'Predicted Fatal value on {prediction_date} is : {prediction:.0f}')

if __name__ == '__main__':
    main()
