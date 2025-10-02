import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram

# Configuração para plots mais bonitos
sns.set_theme(style="whitegrid")
def plot_periodogram(ts, time_unit, sampling_rate_hz, 
                     fundamental_period_days=52.0, 
                     num_harmonics=10):
    """
    Calcula e plota o periodograma de uma série temporal.
    
    Args:
        ts (pd.Series or np.ndarray): A série temporal.
        time_unit (str): A unidade de tempo para os títulos ('Diária', 'Semanal', 'Mensal').
        sampling_rate_hz (float): A taxa de amostragem em amostras por dia.
        fundamental_period_days (float): Período fundamental para análise dos harmônicos.
        num_harmonics (int): Número de harmônicos a plotar.
    """
    # Garante que ts é um array numpy para a função periodogram
    if isinstance(ts, pd.Series):
        ts_values = ts.values
    else:
        ts_values = ts

    # Calcula o periodograma
    frequencies, power_density = periodogram(ts_values, fs=sampling_rate_hz)
    
    # Prepara o plot
    plt.figure(figsize=(14, 7))
    plt.semilogy(frequencies, power_density, color='darkblue', alpha=0.8, label='Densidade de Potência')
    
    plt.title(f'Periodograma com Harmônicos - Ilha Solteira - Agregação {time_unit}', fontsize=16)
    plt.xlabel('Frequência (Ciclos / Semana)', fontsize=12)
    plt.ylabel('Densidade Espectral de Potência (escala log)', fontsize=12)

    # --- Lógica para Plotar Harmônicos ---
    f_fundamental = 1 / fundamental_period_days
    
    print(f"\nAnalisando frequências para agregação {time_unit}:")
    for k in range(1, num_harmonics + 1):
        freq_k = k * f_fundamental
        period_k = 1 / freq_k
        
        if k == 1:
            color, style, width = 'red', '--', 2.0
            label = f'Fundamental k={k} ({period_k:.1f} semanas)'
        else:
            color, style, width = 'gray', ':', 1.5
            label = f'Harmônico k={k} ({period_k:.1f} semanas)'
            
        plt.axvline(x=freq_k, color=color, linestyle=style, linewidth=width, label=label)
        print(f" -> Harmônico k={k}: Frequência = {freq_k:.5f} (Período = {period_k:.1f} semanas)")

    # Ajusta o zoom do eixo X para focar na área de interesse
    plt.xlim(0, (num_harmonics + 1) * f_fundamental)
    
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.show()

inflow, n, T = utils.load_data("SOBRADINHO", "dia")

plt.figure(figsize=(12,8))
time = np.arange(n)
plt.plot(time, inflow, "-", linewidth=1, color="black", label="Afluência")
plt.title("Afluência (m^3/s) em Sobradinho - Agrupamento Diário")
plt.xlabel("Dias desde 01/01/2021")
plt.ylabel("Afluência (m^3/s)")
plt.legend()
plt.tight_layout()
#plt.savefig("img/inflow_sobradinho_dia.png")
plt.show()