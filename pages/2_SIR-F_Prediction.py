import streamlit as st
import torch
import numpy as np
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D
from neurodiffeq.conditions import IVP
from neurodiffeq.networks import FCNN
from neurodiffeq.monitors import Monitor1D
from matplotlib import pyplot as plt
from time import process_time
from scipy.optimize import fsolve

def implicit_euler_residual(yp, ode, to, yo, tp):
    """
    Evaluates the residual of the implicit Euler.
    :param yp: Estimated solution value at the new time
    :param ode: The right hand side of the ODE
    :param to: The old time
    :param yo: The old solution value
    :param tp: The new time
    :return: The residual
    """
    value = yp - yo - (tp - to) * ode(tp, yp)
    return value

def implicit_euler(ode, y0, tspan, num_steps=10):
    """
    Numerical approximation of the solution to the SIRF system using implicit Euler method.
    :param ode: The right hand side of the ODE
    :param y0: Initial conditions
    :param tspan: Time span [t0, t1]
    :param num_steps: Number of time steps
    :return: Solutions for S, I, R, F and the corresponding time points
    """
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)

    t = np.zeros(num_steps + 1)
    y = np.zeros([num_steps + 1, m])
    dt = (tspan[1] - tspan[0]) / float(num_steps)

    t[0] = tspan[0]
    y[0, :] = y0

    for i in range(0, num_steps):
        to = t[i]
        yo = y[i, :]
        tp = t[i] + dt
        yp = yo + dt * ode(to, yo)
        yp = fsolve(lambda yp: implicit_euler_residual(yp, ode, to, yo, tp), yp)
        t[i + 1] = tp
        y[i + 1, :] = yp[:]

    return y, t

def sirf_deriv(t, values):
    """
    Derivative of SIRF system equations.
    :param t: Input value
    :param values: The initial conditions
    :return: The derivatives of the equations
    """
    s = values[0]
    i = values[1]
    r = values[2]
    f = values[3]

    beta = 0.615
    gamma = 0.193
    alpha1 = 0.06
    alpha2 = 0.03
    N = 340000000

    dsdt = - beta * s * i / N
    didt = - (1 - alpha1) * beta * s * i / N - (gamma + alpha2) * i
    drdt = gamma * i
    dfdt = alpha1 * beta * s * i / N - alpha2 * i

    derivs = np.array([dsdt, didt, drdt, dfdt])
    return derivs

def neural_network(epochs, neurons):
    """
    Solves the SIRF system of equations using a fully connected neural network with sigmoid activation.
    :param epochs: Number of epochs for training
    :param neurons: Number of neurons in each hidden layer
    :return: The solutions to the system
    """
    beta, gamma, alpha1, alpha2, N = 0.615, 0.193, 0.06, 0.03, 340000000

    sirf = lambda s, i, r, f, t : [
        diff(s, t) + (- beta * s * i / N),
        diff(i, t) + (- (1 - alpha1) * beta * s * i / N - (gamma + alpha2) * i),
        diff(r, t) - (gamma * i),
        diff(f, t) - (alpha1 * beta * s * i / N - alpha2 * i)
    ]

    # Initial conditions.
    init_vals_sirf = [
        IVP(t_0=0.0, u_0=10.0),
        IVP(t_0=0.0, u_0=1.0),
        IVP(t_0=0.0, u_0=0.0),
        IVP(t_0=0.0, u_0=0.0)
    ]

    # Sets up the neural network with sigmoid activation.
    nets_sirf = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(neurons, neurons), actv=torch.nn.Sigmoid),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(neurons, neurons), actv=torch.nn.Sigmoid),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(neurons, neurons), actv=torch.nn.Sigmoid),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(neurons, neurons), actv=torch.nn.Sigmoid)
    ]

    monitor = Monitor1D(t_min=0.0, t_max=25.0, check_every=100)
    monitor_callback = monitor.to_callback()

    solver = Solver1D(
        ode_system=sirf,
        conditions=init_vals_sirf,
        t_min=0.1,
        t_max=25.0,
        nets=nets_sirf
    )

    solver.fit(max_epochs=epochs, callbacks=[monitor_callback])
    solution_sirf = solver.get_solution()
    ts = np.linspace(0, 25, 100)
    s_net, i_net, r_net, f_net = solution_sirf(ts, to_numpy=True)
    return ts, s_net, i_net, r_net, f_net

def plot_results(ts, s_net, i_net, r_net, f_net, s_num, t):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(ts, s_net, label='NN Susceptible')
    plt.plot(ts, i_net, label='NN Infected')
    plt.plot(ts, r_net, label='NN Recovered')
    plt.plot(ts, f_net, label='NN Fatal')
    plt.plot(t, s_num, '--', label='NUM Susceptible')
    plt.plot(t, s_num, '--', label='NUM Infected')
    plt.plot(t, s_num, '--', label='NUM Recovered')
    plt.plot(t, s_num, '--', label='NUM Fatal')
    plt.legend()
    plt.title('Approximated solutions to the SIRF System')
    plt.xlabel('Time')
    plt.ylabel('Population')

    plt.subplot(2, 1, 2)
    plt.plot(t, s_num)
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Fatal'])
    plt.title('Numerical Approximation')
    plt.xlabel('Time')
    plt.ylabel('Population')

    plt.tight_layout()
    st.pyplot()

def main():
    st.title('SIR-F System Prediction using Neural Networks')

    st.sidebar.header('Configuration')
    epochs = st.sidebar.slider('Number of Epochs', min_value=100, max_value=5000, value=1000, step=100)
    neurons = st.sidebar.slider('Number of Neurons', min_value=16, max_value=128, value=32, step=16)

    st.sidebar.text('Training Configuration:')
    st.sidebar.text(f'- Number of Epochs: {epochs}')
    st.sidebar.text(f'- Number of Neurons: {neurons}')

    st.sidebar.header('Training Progress')

    if st.sidebar.button('Train Model'):
        st.sidebar.text('Training in progress...')

        # Neural network approximation.
        t1_start = process_time()
        ts, s_net, i_net, r_net, f_net = neural_network(epochs, neurons)
        t1_stop = process_time()

        st.sidebar.text(f'Training completed in {t1_stop - t1_start:.2f} seconds.')

        st.sidebar.text('Generating plots...')
        tspan = np.array([0.0, 25])
        y0 = np.array([10, 1, 0, 0])
        n = 100

        t2_start = process_time()
        s_num, t = implicit_euler(sirf_deriv, y0, tspan, n)
        t2_stop = process_time()

        plot_results(ts, s_net, i_net, r_net, f_net, s_num, t)

        st.sidebar.text(f'Elapsed time for numerical approximation: {t2_stop - t2_start:.2f} seconds.')

if __name__ == '__main__':
    main()
