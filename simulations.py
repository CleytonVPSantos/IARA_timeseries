import numpy as np
import matplotlib.pyplot as plt
import utils

phi_1 = 0.8998
sigma_1 = 0.4527
T = 52
delta_1 = 1
change_probability = 1/500
total_states = 500
delta_2 = (T/total_states)*change_probability
init = 0
years = 24
inflow_fourier_coef = [1200.24,682.04,694.54,240.92,71.95,-78.27,74.44]
std_fourier_coef = [565.81,370.63,414.39,102.59,-15.44,-65.50,3.96]
total_steps = int(years*total_states*(1/change_probability))
harmonics = 3

def simple_ar_1_future(phi, sigma, x0, total_forecast_steps):
    current_value = x0
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
        else:
            states.append(current_state)
        
    return states


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
    phi_2 = phi_1**(delta_2 / delta_1)
    sigma_2 = np.sqrt((1 - phi_2**2) / (1 - phi_1**2)) * sigma_1

    simulated_values = simple_ar_1_future(phi_2, sigma_2, init, total_steps)
    visited_states = np.array(random_walk(change_probability, total_states, total_steps), dtype=int)

    fourier_inflow, fourier_std = sazonality(
        total_states, harmonics, inflow_fourier_coef, std_fourier_coef, T, delta_2/change_probability
    )
    
    sazonal_inflow = fourier_inflow[visited_states]
    std_sazonal = fourier_std[visited_states]
    
    inflow_predictions = sazonal_inflow + std_sazonal * simulated_values

    time = np.arange(total_steps)
    plt.figure(figsize=(15, 7))
    plt.plot(time, inflow_predictions, '-', linewidth=1, color="blue", label="Afluência Prevista")

    plt.plot(time, sazonal_inflow, '--', linewidth=1.5, color="red", label="Componente Sazonal (baseada no estado)")
    plt.title("Simulação da Afluência por Estados Semanais")
    plt.xlabel("Passos da Simulação")
    plt.ylabel("Afluência")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


main()
                          