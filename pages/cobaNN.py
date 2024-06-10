import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

def train_model(df, num_hidden_layers, num_neurons, epochs):
    # Feature engineering and preprocessing
    X = df[['susceptible', 'infected', 'recovered', 'fatal', 'total population']]
    y = df['confirmed']
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Define neural network model
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=X.shape[1], activation='relu'))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train model
    model.fit(X_scaled, y, epochs=epochs, batch_size=10, verbose=0)
    return model, scaler

def predict_confirmed_cases(model, scaler, susceptible, infected, recovered, fatal, total_population):
    # Scale the input data
    new_data = np.array([[susceptible, infected, recovered, fatal, total_population]])
    new_data_scaled = scaler.transform(new_data)
    # Perform prediction
    predicted_confirmed_cases = model.predict(new_data_scaled)
    return predicted_confirmed_cases[0][0]

def main():
    st.title('COVID-19 Confirmed Cases Prediction')

    # Input form for new data
    susceptible = st.number_input('Susceptible', value=100000)
    infected = st.number_input('Infected', value=1000)
    recovered = st.number_input('Recovered', value=500)
    fatal = st.number_input('Fatal', value=50)
    total_population = st.number_input('Total Population', value=500000)

    # Input form for model parameters
    num_hidden_layers = st.number_input('Number of Hidden Layers', value=2, min_value=1)
    num_neurons = st.number_input('Number of Neurons per Layer', value=64, min_value=1)
    epochs = st.number_input('Number of Epochs', value=50, min_value=1)

    # Read the CSV file
    file_path = 'cases_malaysia.csv'
    df = pd.read_csv(file_path)

    # Train model
    model, scaler = train_model(df, num_hidden_layers, num_neurons, epochs)

    # Predict button
    if st.button('Predict Confirmed Cases'):
        # Perform prediction
        prediction = predict_confirmed_cases(model, scaler, susceptible, infected, recovered, fatal, total_population)
        # Display prediction
        st.success(f'Predicted Confirmed Cases: {prediction:.2f}')

if __name__ == '__main__':
    main()
