import numpy as np
import matplotlib.pyplot as plt
import utils
from dataclasses import dataclass
import pandas as pd

HISTORIC_YEARS = 24

@dataclass
class ModelParameters:
    posto: str
    phi_1: float
    sigma_1: float
    inflow_fourier_coef: list[float]
    std_fourier_coef: list[float]

# load parameters for simulation
def load_parameters(posto_nome: str, caminho_csv: str) -> ModelParameters:
    print(f"Carregando parâmetros para o posto: {posto_nome}...")
    try:
        params_df = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado em: {caminho_csv}")

    posto_row = params_df[params_df['posto'] == posto_nome]

    if posto_row.empty:
        raise ValueError(f"Posto '{posto_nome}' não foi encontrado no arquivo de parâmetros.")

    phi_1 = posto_row['ar_coef_1'].iloc[0]
    sigma_1 = posto_row['residuals_std'].iloc[0]

    inflow_coefs = [posto_row['inflow_fourier_coef_1'].iloc[0]] + \
                   [posto_row[f'inflow_fourier_coef_{i}'].iloc[0] for i in range(2, 8)]
    
    std_coefs = [posto_row['std_fourier_coef_1'].iloc[0]] + \
                [posto_row[f'std_fourier_coef_{i}'].iloc[0] for i in range(2, 8)]

    return ModelParameters(
        posto=posto_nome,
        phi_1=phi_1,
        sigma_1=sigma_1,
        inflow_fourier_coef=inflow_coefs,
        std_fourier_coef=std_coefs
    )

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
def seasonality_mean(total_states, harmonics, inflow_fourier_coef, std_fourier_coef, T, step_size):
    k = np.arange(1, harmonics + 1).reshape(-1, 1)

    s_start = np.arange(total_states).reshape(1, -1)
    s_end = np.arange(1, total_states + 1).reshape(1, -1)

    omega_k = 2 * k * np.pi / T
    
    t1 = s_start * step_size
    t2 = s_end * step_size

    A_inflow = np.array(inflow_fourier_coef[1::2])[:harmonics].reshape(-1, 1)
    B_inflow = np.array(inflow_fourier_coef[2::2])[:harmonics].reshape(-1, 1)

    integrated_harmonics_inflow = (A_inflow / omega_k) * (np.sin(omega_k * t2) - np.sin(omega_k * t1)) \
                                - (B_inflow / omega_k) * (np.cos(omega_k * t2) - np.cos(omega_k * t1))

    fourier_inflow = inflow_fourier_coef[0] + np.sum(integrated_harmonics_inflow, axis=0) / step_size

    A_std = np.array(std_fourier_coef[1::2])[:harmonics].reshape(-1, 1)
    B_std = np.array(std_fourier_coef[2::2])[:harmonics].reshape(-1, 1)

    integrated_harmonics_std = (A_std / omega_k) * (np.sin(omega_k * t2) - np.sin(omega_k * t1)) \
                             - (B_std / omega_k) * (np.cos(omega_k * t2) - np.cos(omega_k * t1))
    
    fourier_std = std_fourier_coef[0] + np.sum(integrated_harmonics_std, axis=0) / step_size

    return fourier_inflow, fourier_std

def seasonality_point(total_states, harmonics, inflow_fourier_coef, std_fourier_coef, T, step_size):
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

def exec_simulation(params, total_states, i, change_probability, years, posto):

    print(f"\nExecutando simulação para: Posto='{params.posto}', Estados={total_states}, Prob. Mudança={change_probability:.2f}")

    historic_inflow, n, T = utils.load_data(posto, "sem")
    delta_1 = 1
    init = 0
    harmonics = 3 

    delta_2 = (T / total_states) * change_probability
    total_steps = int(years * total_states * (1 / change_probability))
    phi_2 = params.phi_1 ** (delta_2 / delta_1)
    sigma_2 = np.sqrt((1 - phi_2**2) / (1 - params.phi_1**2)) * params.sigma_1

    simulated_values = simple_ar_1_future(phi_2, sigma_2, init, total_steps)
    visited_states = random_walk(change_probability, total_states, total_steps)

    fourier_inflow, fourier_std = seasonality_point(
        total_states, harmonics, params.inflow_fourier_coef, params.std_fourier_coef, T, delta_2 / change_probability
    )
    
    sazonal_inflow = fourier_inflow[visited_states] 
    std_sazonal = fourier_std[visited_states]
    inflow_predictions = sazonal_inflow + std_sazonal * simulated_values

    print("Simulação concluída. Gerando gráficos...")
    for period in [int(k*total_states/2) for k in range(2)]:
        utils.sample_vs_normal(inflow_predictions[visited_states == period], 
                                inflow_predictions[visited_states == period].size,
                                "simul/"+posto+"hisp"+str(i)+"t"+str(total_states)+"t"+str(period))

        utils.compare_histogram_vertical(historic_inflow.reshape(HISTORIC_YEARS, T)[:,int(period*T/total_states):int((period+1)*T/total_states)].flatten(), 
        inflow_predictions[visited_states == period],
        "simul/"+posto+"comp"+str(i)+"t"+str(total_states)+"t"+str(period))

        utils.plot([inflow_predictions[:int(total_states/change_probability*6)], sazonal_inflow[:int(total_states/change_probability*6)]], 
                total_states/change_probability*6, 
                ['-', '--'], 
                ["blue", "red"],
                ["Vazão predita", "Componente sazonal (baseada no estágio)"],
                "Simulação da vazão",
                "Passo da simulação",
                    "Inflow",
                    "simul/"+posto+"simp"+str(i)+"t"+str(total_states))

def main():
    POSTO_ALVO = "SOBRADINHO" 
    CAMINHO_PARAMETROS_CSV = "results/model3_sem.csv" 

    total_states_list = [5,50,500]   
    probabilities = [1.0, 1/10, 1/100]
    simulation_years = 1000

    try:
        parameters = load_parameters(POSTO_ALVO, CAMINHO_PARAMETROS_CSV)
    except (FileNotFoundError, ValueError) as e:
        print(f"Critical error: {e}")
        return 

    for states in total_states_list:
        for i, prob in enumerate(probabilities):
            exec_simulation(parameters, states, i, prob, simulation_years, POSTO_ALVO)


if __name__ == "__main__":
    main()