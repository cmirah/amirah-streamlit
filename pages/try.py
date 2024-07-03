import streamlit as st
import torch
from neurodiffeq import diff
from neurodiffeq.ode import Solver1D, IVP
from neurodiffeq.networks import FCNN
import numpy as np
import matplotlib.pyplot as plt

# Define the SIR-F model
def sir_f_ode(t, S, I, R, F, beta=0.3, gamma=0.1, mu=0.01):
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I - mu * I
    dR_dt = gamma * I
    dF_dt = mu * I
    return [dS_dt, dI_dt, dR_dt, dF_dt]

# Accept user inputs
epochs = st.number_input('Enter the number of epochs:', min_value=1, value=100)
neurons = st.number_input('Enter the number of neurons:', min_value=1, value=50)
time = st.number_input('Enter the time period for prediction:', min_value=1, value=160)

# Define and train the neural network
net_s = FCNN(n_input_units=1, n_output_units=1, n_hidden_units=neurons, n_hidden_layers=2, actv=torch.nn.Tanh)
net_i = FCNN(n_input_units=1, n_output_units=1, n_hidden_units=neurons, n_hidden_layers=2, actv=torch.nn.Tanh)
net_r = FCNN(n_input_units=1, n_output_units=1, n_hidden_units=neurons, n_hidden_layers=2, actv=torch.nn.Tanh)
net_f = FCNN(n_input_units=1, n_output_units=1, n_hidden_units=neurons, n_hidden_layers=2, actv=torch.nn.Tanh)

# Initial conditions for S, I, R, F
initial_conditions = [
    IVP(t_0=0.0, x_0=0.99),  # Initial condition for S
    IVP(t_0=0.0, x_0=0.01),  # Initial condition for I
    IVP(t_0=0.0, x_0=0.0),   # Initial condition for R
    IVP(t_0=0.0, x_0=0.0)    # Initial condition for F
]

solver = Solver1D(
    ode_system=sir_f_ode,
    conditions=initial_conditions,
    t_min=0.0,
    t_max=time,
    nets=[net_s, net_i, net_r, net_f]
)

# Train the network
solver.fit(max_epochs=epochs)

# Make predictions
ts = torch.linspace(0, time, 100).reshape(-1, 1)

# Get predictions
s_net, i_net, r_net, f_net = solver.get_solution(ts, as_type='np')

# Display results
st.write("Predicted S(t):", s_net)
st.write("Predicted I(t):", i_net)
st.write("Predicted R(t):", r_net)
st.write("Predicted F(t):", f_net)

# Optionally, plot the results
plt.figure(figsize=(10, 6))
plt.plot(ts, s_net, label='S(t)')
plt.plot(ts, i_net, label='I(t)')
plt.plot(ts, r_net, label='R(t)')
plt.plot(ts, f_net, label='F(t)')
plt.xlabel('Time')
plt.ylabel('Proportions')
plt.title('SIR-F Model Predictions')
plt.legend()
st.pyplot(plt)

