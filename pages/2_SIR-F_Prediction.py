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

st.image("cvd.png", width = 500)

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

def neural_network(epochs, neurons, show_progress=False):
    """
    Solves the SIRF system of equations using a fully connected neural network with sigmoid activation.
    :param epochs: Number of epochs for training
    :param neurons: Number of neurons in each hidden layer
    :param show_progress: Whether to show training progress
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

    if show_progress:
        st.text('Training in progress...')
    
    solver.fit(max_epochs=epochs, callbacks=[monitor_callback])

    if show_progress:
        st.text('Training completed.')

    solution_sirf = solver.get_solution()
    ts = np.linspace(0, 25, 100)
    s_net, i_net, r_net, f_net = solution_sirf(ts, to_numpy=True)
    return ts, s_net, i_net, r_net, f_net

def plot_results(ts, s_net, i_net, r_net, f_net, s_num, t):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(ts, s_net, label='NN Susceptible')
    axs[0].plot(ts, i_net, label='NN Infected')
    axs[0].plot(ts, r_net, label='NN Recovered')
    axs[0].plot(ts, f_net, label='NN Fatal')
    axs[0].plot(t, s_num, '--', label='NUM Susceptible')
    axs[0].plot(t, s_num, '--', label='NUM Infected')
    axs[0].plot(t, s_num, '--', label='NUM Recovered')
    axs[0].plot(t, s_num, '--', label='NUM Fatal')
    axs[0].legend()
    axs[0].set_title('Approximated solutions to the SIRF System')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Population')

    axs[1].plot(t, s_num)
    axs[1].legend(['Susceptible', 'Infected', 'Recovered', 'Fatal'])
    axs[1].set_title('Numerical Approximation')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Population')

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title('SIR-F System Prediction using Neural Networks')

    st.sidebar.header('Configuration')
    epochs = st.sidebar.slider('Number of Epochs', min_value=100, max_value=5000, value=1000, step=100)
    neurons = st.sidebar.slider('Number of Neurons', min_value=16, max_value=128, value=32, step=16)

    st.sidebar.text('Training Configuration:')
    st.sidebar.text(f'- Number of Epochs: {epochs}')
    st.sidebar.text(f'- Number of Neurons: {neurons}')

    if st.sidebar.button('Train Model'):
        ts, s_net, i_net, r_net, f_net = neural_network(epochs, neurons, show_progress=True)
        
        st.text('Training completed.')

        st.text('Generating plots...')
        tspan = np.array([0.0, 25])
        y0 = np.array([10, 1, 0, 0])
        n = 100

        s_num, t = implicit_euler(sirf_deriv, y0, tspan, n)

        plot_results(ts, s_net, i_net, r_net, f_net, s_num, t)

        st.text('Elapsed time for numerical approximation: {:.2f} seconds.'.format(process_time()))

if __name__ == '__main__':
    main()


