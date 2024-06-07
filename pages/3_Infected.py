import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="I_Prediction",page_icon="📘")
st.title("Prediction of COVID-19 Disease using SIRF Model")
st.subheader("Infected Prediction")

def train_model(df):
    # Feature engineering and preprocessing
    X = df[['susceptible', 'recovered', 'fatal']]
    y = df['infected']
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestRegressor()
    model.fit(X_scaled, y)
    return model, scaler

def predict_infected(model, scaler, susceptible, recovered, fatal):
    # Scale the input data
    new_data = np.array([[susceptible, recovered, fatal]])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_infected = model.predict(new_data_scaled)
    return predicted_infected[0]

def main():
    # Input form for new data
    susceptible = st.number_input('Susceptible', value=100000)
    recovered = st.number_input('Recovered', value=500)
    fatal = st.number_input('Fatal', value=50)

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)

    # Train model
    model, scaler = train_model(df)

    # Predict button
    if st.button('Predict Infected'):
        # Perform prediction
        prediction = predict_infected(model, scaler, susceptible, recovered, fatal)
        # Display prediction
        st.success(f'Predicted Infected value is : {prediction:.0f}')

if __name__ == '__main__':
    main()