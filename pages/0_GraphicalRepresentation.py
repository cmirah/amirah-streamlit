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

## Total number of population in Malaysia in 2023
totalPopulation = 34308525

## number of days
T = 100

import matplotlib.pyplot as plt
import numpy as np

def modelSIRF(X0, beta, gamma, alpha1, alpha2, T, stepCount):

    def func(X):
        N = np.sum(X)
        f1 = -beta*X[0]*X[1]/N 
        f2 = (1-alpha1)*beta*X[0]*X[1]/N - (gamma+alpha2)*X[1]
        f3 = gamma*X[1] 
        f4 = alpha1*beta*X[0]*X[1]/N - alpha2*X[1]
        return np.array([f1, f2, f3, f4])

    h = T / stepCount
    Xvals = [list(X0)]

    X = X0.copy()
    for n in range(stepCount):
        k1 = func(X)
        k2 = func(X + h*k1/2.0)
        k3 = func(X + h*k2/2.0)
        k4 = func(X + h*k3)

        X += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        Xvals.append(list(X))                   

    return np.stack(Xvals)


## display the model
def plotSIRF(Xvals, ax, title, time=T, show=[0,1,2,3,4]):
    stepCount = Xvals.shape[0] - 1
    t = np.linspace(0, time, stepCount+1)
    PopTot = np.sum(Xvals, axis=1)

    if 0 in show or show is None:
        ax.plot(t, Xvals[:,0], label="Susceptibles", c='g')
    if 1 in show or show is None:
        ax.plot(t, Xvals[:,1], label="Infected", c='r')
    if 2 in show or show is None:
        ax.plot(t, Xvals[:,2], label="Recovered", c='b')
    if 3 in show or show is None:
        ax.plot(t, Xvals[:,3], label="Fatal", c='orange')
    if 4 in show or show is None:
        ax.plot(t, PopTot, label="Total Population", c="k")

    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title(title, y=1.02, size="xx-large")

    ax.legend()
    plt.tight_layout()


## Parameters for the SIR-F model
S0, I0, R0, F0 = totalPopulation-3e5, 3e5, 0, 0
beta = 0.615
gamma = 0.193
alpha1 = 0.06
alpha2 = 0.03
stepCount = 1000  # Define the stepCount here

## Simulation over 100 days
Time = 100

X0 = np.array([S0, I0, R0, F0])
modelParams = {"X0":X0, "beta":beta, "gamma":gamma, "alpha1":alpha1, "alpha2":alpha2, "T":Time, "stepCount":stepCount}

Xvals = modelSIRF(**modelParams)
# print(Xvals.shape)

fig, ax = plt.subplots(1, 1, figsize=(10,6))
plotSIRF(Xvals, ax, time=Time, title="Illustration of the SIR-F model")

# plt.savefig("Images/Simu1.png")
