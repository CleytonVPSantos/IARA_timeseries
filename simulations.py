import numpy as np
import matplotlib.pyplot as plt
import utils

# parameters
phi_1 = 0.8998 # AR coefficients
sigma_1 = 0.4527 # residuals std
T = 52 # number of periods considered for the AR/Fourier fit
delta_1 = 1 # stardard
change_probability = 1/10 # probability of going to the next state
total_states = 5 # number of states in the discretization
delta_2 = (T/total_states)*change_probability # new delta
init = 0 # inital value for the AR
years = 10000
inflow_fourier_coef = [1200.24,682.04,694.54,240.92,71.95,-78.27,74.44] # inflow coefficients
std_fourier_coef = [565.81,370.63,414.39,102.59,-15.44,-65.50,3.96] # coefficients of fourier approx. of std
total_steps = int(years*total_states*(1/change_probability)) # total number of steps for simulation
harmonics = 3 # number of harmonics used in fourier approximations

# generates a AR process (independent of state)
def simple_ar_1_future(phi, sigma, x0, total_forecast_steps):
    current_value = x0
    simulated_values = []

    for _ in range(total_forecast_steps):
        next_val = phi * current_value + np.random.normal(0, sigma)
        simulated_values.append(next_val)
        current_value = next_val

    return np.array(simulated_values)

# generates a random walk in the states graph
def random_walk(change_probability, total_states, total_steps):
    stay_probability = 1 - change_probability
    advances = np.random.choice([0, 1], size=total_steps, p=[stay_probability, change_probability])
    advances[0] = 0 
    
    states = np.cumsum(advances) % total_states
    
    return states

# calculates seazonal components for each state
def sazonality(total_states, harmonics, inflow_fourier_coef, std_fourier_coef, T, step_size):
    k = np.arange(1, harmonics + 1).reshape(-1, 1)
    s = np.arange(total_states).reshape(1, -1)   

    angle = 2 * k * np.pi * s * step_size / T
    
    A_inflow = np.array(inflow_fourier_coef[1::2])[:harmonics].reshape(-1, 1)
    B_inflow = np.array(inflow_fourier_coef[2::2])[:harmonics].reshape(-1, 1)
    harmonics_matrix_inflow = A_inflow * np.cos(angle) + B_inflow * np.sin(angle)
    fourier_inflow = inflow_fourier_coef[0] + np.sum(harmonics_matrix_inflow, axis=0)
    
    A_std = np.array(std_fourier_coef[1::2])[:harmonics].reshape(-1, 1)
    B_std = np.array(std_fourier_coef[2::2])[:harmonics].reshape(-1, 1)
    harmonics_matrix_std = A_std * np.cos(angle) + B_std * np.sin(angle)
    fourier_std = std_fourier_coef[0] + np.sum(harmonics_matrix_std, axis=0)

    return fourier_inflow, fourier_std


def main():
    total_states_list = [5, 50, 500]
    probabilities = [1, 1/10, 1/100]
    inflow, n, T0 = utils.load_data("SOBRADINHO", "sem")
    for total_states in total_states_list:
        for i, change_probability in enumerate(probabilities):
            years = 100000*change_probability
            delta_2 = (T/total_states)*change_probability # new delta
            total_steps = int(years*total_states*(1/change_probability)) # total number of steps for simulation
            phi_2 = phi_1**(delta_2 / delta_1) # new phi
            sigma_2 = np.sqrt((1 - phi_2**2) / (1 - phi_1**2)) * sigma_1 # new sigma

            simulated_values = simple_ar_1_future(phi_2, sigma_2, init, total_steps)
            visited_states = np.array(random_walk(change_probability, total_states, total_steps), dtype=int)

            fourier_inflow, fourier_std = sazonality(
                total_states, harmonics, inflow_fourier_coef, std_fourier_coef, T, delta_2/change_probability
            )
            
            sazonal_inflow = fourier_inflow[visited_states] 
            std_sazonal = fourier_std[visited_states]
            inflow_predictions = sazonal_inflow + std_sazonal * simulated_values
 
            for period in [int(k*total_states/2) for k in range(2)]:
                utils.sample_vs_normal(inflow_predictions[visited_states == period], 
                                       inflow_predictions[visited_states == period].size,
                                       "simul/hisp"+str(i)+"t"+str(total_states)+"t"+str(period))

                utils.compare_histogram_vertical(inflow.reshape(24, T0)[:,int(period*T/total_states):int((period+1)*T/total_states)].flatten(), 
                inflow_predictions[visited_states == period],
                "simul/comp"+str(i)+"t"+str(total_states)+"t"+str(period))
            utils.plot([inflow_predictions[:int(total_states/change_probability*6)], sazonal_inflow[:int(total_states/change_probability*6)]], 
                    total_states/change_probability*6, 
                    ['-', '--'], 
                    ["blue", "red"],
                    ["Inflow forecast", "Sazonal component (based on stage)"],
                    "Inflow simulation",
                    "Simulation steps",
                        "Inflow",
                        "simul/simp"+str(i)+"t"+str(total_states))

main()