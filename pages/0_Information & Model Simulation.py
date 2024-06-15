# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_icon="📈")

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select Page', ['Information', 'Model Simulation'])

if page == 'Information':
    # Information Section
    st.title("Machine Learning Workflow")
    st.image("11.png", width=500)
    st.title("The basic SIR model and the derived SIR-F")
    st.subheader("Mathematical Model of SIR-F")
    
    st.write("1) SIR-F Model:")
    st.markdown("""
    - **S: Susceptible** = (Total Population - Confirmed)
    - **I: Infected** = (Confirmed - Recovered - Fatal)
    - **R: Recovered**
    - **F: Fatal**
    """)
    
    st.write("2) Model Relationship:")
    st.latex(r"S \rightarrow \beta I \rightarrow \gamma R , I \rightarrow \alpha F")
    st.write("where the parameters used are:")
    st.markdown("""
    - **α : Mortality rate**
    - **β : Effective contact rate**
    - **γ : Recovery rate**
    """)
    
    st.write("3)")
    st.image("1.png", width=500)
    st.write("where: N is the total population & t is the elapsed time from the start date.")
    
    st.write("4)")
    st.image("2.png", width=500)
    st.write("where: α1 = mortality rate of S*, α2 = mortality rate of I and N = S + I + R + F.")
    
    st.markdown("""
    **👈 Select a button from the sidebar** to see the simulation graph of the models!
    """)


def modelSIRF(X0, beta, gamma, alpha1, alpha2, T, stepCount):
    def func(X):
        N = np.sum(X)
        f1 = -beta * X[0] * X[1] / N 
        f2 = (1 - alpha1) * beta * X[0] * X[1] / N - (gamma + alpha2) * X[1]
        f3 = gamma * X[1]
        f4 = alpha1 * beta * X[0] * X[1] / N - alpha2 * X[1]
        return np.array([f1, f2, f3, f4])

    h = T / stepCount
    Xvals = [list(X0)]
    X = X0.copy().astype(float)
    for n in range(stepCount):
        k1 = func(X)
        k2 = func(X + h * k1 / 2.0)
        k3 = func(X + h * k2 / 2.0)
        k4 = func(X + h * k3)
        X += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        Xvals.append(list(X))

    return np.stack(Xvals)

def plotSIRF(Xvals, ax, title, time=None, show=None):
    stepCount = Xvals.shape[0] - 1
    t = np.linspace(0, time, stepCount + 1)
    PopTot = np.sum(Xvals, axis=1)

    if show is None:
        show = [0, 1, 2, 3, 4]

    if 0 in show:
        ax.plot(t, Xvals[:, 0], label="Susceptibles", c='g')
    if 1 in show:
        ax.plot(t, Xvals[:, 1], label="Infected", c='r')
    if 2 in show:
        ax.plot(t, Xvals[:, 2], label="Recovered", c='b')
    if 3 in show:
        ax.plot(t, Xvals[:, 3], label="Fatal", c='orange')
    if 4 in show:
        ax.plot(t, PopTot, label="Total Population", c="k")

    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title(title, y=1.02, size="xx-large")
    ax.legend()
    plt.tight_layout()

if page == 'Model Simulation':
    def main():
        st.title("SIR-F Model Simulation")

        # Simulation Parameters Section
        st.sidebar.header("Simulation Parameters")
        st.sidebar.markdown("(Values can be adjusted)")
        # Simulation Parameters Section
        st.sidebar.header("Simulation Parameters")
        st.sidebar.markdown("(Values can be adjusted)")
        T = st.sidebar.slider("Total Number of Days", min_value=1, max_value=365, value=100)
        S0 = st.sidebar.slider("Initial Susceptible Population", min_value=0, max_value=50000000, value=34000000, step=1000000)
        I0 = st.sidebar.slider("Initial Infected Population", min_value=0, max_value=1000000, value=300000, step=10000)
        R0 = st.sidebar.slider("Initial Recovered Population", min_value=0, max_value=1000000, value=0, step=10000)
        F0 = st.sidebar.slider("Initial Fatal Population", min_value=0, max_value=1000000, value=0, step=10000)
        beta = st.sidebar.slider("Transmission Rate (β)", min_value=0.0, max_value=1.0, value=0.615)
        gamma = st.sidebar.slider("Recovery Rate (γ)", min_value=0.0, max_value=1.0, value=0.193)
        alpha1 = st.sidebar.slider("Fatal Recovery Rate (α1)", min_value=0.0, max_value=1.0, value=0.06)
        alpha2 = st.sidebar.slider("Fatal Transmission Rate (α2)", min_value=0.0, max_value=1.0, value=0.03)
        stepCount = st.sidebar.slider("Number of Steps", min_value=100, max_value=10000, value=1000, step=100)


        X0 = np.array([S0, I0, R0, F0])
        modelParams = {"X0": X0, "beta": beta, "gamma": gamma, "alpha1": alpha1, "alpha2": alpha2, "T": T, "stepCount": stepCount}

        Xvals = modelSIRF(**modelParams)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plotSIRF(Xvals, ax, time=T, title="SIR-F Model Simulation")

        st.pyplot(fig)

    if __name__ == "__main__":
        main()
