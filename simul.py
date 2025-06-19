import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

phi_1 = 0.8998
sigma_1 = 0.4527
T = 52
delta_1 = 1/T
delta = 1/5
init = 0
total_states = 5
total_steps = 1000
harmonics = 3

def simple_ar_1_future(phi, sigma, x0, total_forecast_steps):
    current_value= x0
    simulated_values = []

    for _ in range(total_forecast_steps):
        next_val = phi * current_value + np.random.normal(0, sigma)
        simulated_values.append(next_val)
        current_value = next_val

    return np.array(simulated_values)

def random_walk(change_probability, total_states, total_steps):
    states = [0]
    current_state = 0
    for i in range(total_steps - 1):
        if(change_probability > np.random.rand()):
            new_state = (current_state + 1) % total_states
            states.append(new_state)
            current_state = new_state

    return np.array(states)

def main():
    phi = phi**(delta / delta_1)
    sigma = ((1 - phi**2) / (1 - phi_1**2)) * sigma_1
    simulated_values = simple_ar_1_future(phi, sigma, init, total_steps)
    visited_states = random_walk(1, total_states, total_steps)

    features = [np.ones_like(t)]
    for i in range(1, harmonics + 1):
        features.append(np.sin(2*np.pi*i*t/T))
        features.append(np.cos(2*np.pi*i*t/T))
main()
