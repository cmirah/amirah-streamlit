import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scipy.integrate import odeint

# SIR-F Model
def sirf_model(y, t, N, beta, gamma, alpha1, alpha2):
    S, I, R, F = y
    dSdt = -beta * S * I / N
    dIdt = (1 - alpha1) * beta * S * I / N - (gamma + alpha2) * I
    dRdt = gamma * I
    dFdt = alpha1 * beta * S * I / N - alpha2 * I
    return dSdt, dIdt, dRdt, dFdt

def generate_data(t, y0, N, beta, gamma, alpha1, alpha2):
    sol = odeint(sirf_model, y0, t, args=(N, beta, gamma, alpha1, alpha2))
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

# Streamlit interface
import streamlit as st

st.title("SIR-F Model Prediction Using ANN")

# User inputs
beta = st.sidebar.slider("Infection rate (beta)", 0.0, 1.0, 0.615)
gamma = st.sidebar.slider("Recovery rate (gamma)", 0.0, 1.0, 0.193)
alpha1 = st.sidebar.slider("Death rate (alpha1)", 0.0, 1.0, 0.06)
alpha2 = st.sidebar.slider("Fatality rate (alpha2)", 0.0, 1.0, 0.03)
S0 = st.sidebar.number_input("Initial susceptible population (S0)", value=340000000)
I0 = st.sidebar.number_input("Initial infected population (I0)", value=300000)
R0 = st.sidebar.number_input("Initial recovered population (R0)", value=0)
F0 = st.sidebar.number_input("Initial fatal cases (F0)", value=0)
N = S0 + I0 + R0 + F0
days = st.sidebar.slider("Number of days for simulation", 1, 365, 160)

# Time vector
t = np.linspace(0, days, days)

# Generate data
S, I, R, F = generate_data(t, [S0, I0, R0, F0], N, beta, gamma, alpha1, alpha2)

# Prepare data for ANN
data = pd.DataFrame({'S': S, 'I': I, 'R': R, 'F': F, 't': t})
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
X = scaled_data[:, -1].reshape(-1, 1)  # Time as input
y = scaled_data[:, :-1]  # S, I, R, F as output

# Debugging: Print shapes of X and y
st.write("Shape of X:", X.shape)
st.write("Shape of y:", y.shape)

# Define and train ANN model
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X, y, epochs=100, batch_size=10, verbose=0)

# Predict using the trained model
try:
    predictions = model.predict(X)
    # Debugging: Print shape of predictions
    st.write("Shape of predictions:", predictions.shape)

    # Inverse transform the predictions
    predicted_values = scaler.inverse_transform(np.hstack((predictions, X)))

    # Extract predicted S, I, R, F
    predicted_S = predicted_values[:, 0]
    predicted_I = predicted_values[:, 1]
    predicted_R = predicted_values[:, 2]
    predicted_F = predicted_values[:, 3]

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, S, label='Actual S')
    ax.plot(t, predicted_S, label='Predicted S', linestyle='--')
    ax.plot(t, I, label='Actual I')
    ax.plot(t, predicted_I, label='Predicted I', linestyle='--')
    ax.plot(t, R, label='Actual R')
    ax.plot(t, predicted_R, label='Predicted R', linestyle='--')
    ax.plot(t, F, label='Actual F')
    ax.plot(t, predicted_F, label='Predicted F', linestyle='--')
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    ax.set_title('SIR-F Model Prediction Using ANN')
    ax.legend()
    st.pyplot(fig)

    # Display results
    st.subheader("Results")
    st.write("Actual S values:", S)
    st.write("Predicted S values:", predicted_S)
    st.write("Actual I values:", I)
    st.write("Predicted I values:", predicted_I)
    st.write("Actual R values:", R)
    st.write("Predicted R values:", predicted_R)
    st.write("Actual F values:", F)
    st.write("Predicted F values:", predicted_F)

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.write(e)

