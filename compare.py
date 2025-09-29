import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 
def main():
    inflow, n, T = utils.load_data("SOBRADINHO", "sem")
    
        
    time = np.arange(n)

    fourier_inflow1, deseasonalized_inflow, fourier_sq_error, fourier_coef, AIC = utils.inflow_fourier_predict(inflow, n, T, 4)

    plt.plot(time, inflow, ".", linewidth=1, color="black", label="Componente sazonal")
    plt.plot(time, fourier_inflow1, "-", linewidth=1, color="red", label="Componente sazonal")

    x_start, x_end = 0*52, 24*52
    # Adiciona linhas verticais tracejadas e rótulos de ano
    anos = [i for i in range(2001, 2025)]
    plt.ylim((0, 6000))
    for i, ano in enumerate(anos):
        x = x_start + i*52
        plt.axvline(x=x, color="gray", linestyle="--", linewidth=0.8)
        plt.text(x + 2, plt.ylim()[1]*0.79, str(ano), rotation=0,
                verticalalignment="top", fontsize=9, color="gray")

    plt.title("Previsão da afluência - Modelo 4")
    plt.xlabel("Tempo (semanas desde 01/01/2001)")
    plt.ylabel("Afluência (m³/s)")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    """norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
    ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(norm_deseasonalized_inflow, n, T, 2)
    model_prediction = fourier_inflow + fourier_std * ar_prediction
    
    time = np.arange(n)
    x_start, x_end = 20*52, 24*52

    plt.plot(time[x_start:x_end], inflow[x_start:x_end], ".", markersize=4, color="black", label="Dados reais")
    plt.plot(time[x_start:x_end], model_prediction[x_start:x_end], "-", linewidth=1, color="blue", label="Previsão")
    plt.plot(time[x_start:x_end], fourier_inflow[x_start:x_end], "--", linewidth=1, color="red", label="Componente sazonal")

    # Adiciona linhas verticais tracejadas e rótulos de ano
    anos = [i for i in range(2021, 2025)]
    plt.ylim((0, 6000))
    for i, ano in enumerate(anos):
        x = x_start + i*52
        plt.axvline(x=x, color="gray", linestyle="--", linewidth=0.8)
        plt.text(x + 2, plt.ylim()[1]*0.79, str(ano), rotation=0,
                verticalalignment="top", fontsize=9, color="gray")

    plt.title("Previsão da afluência - Modelo 4")
    plt.xlabel("Tempo (semanas desde 01/01/2001)")
    plt.ylabel("Afluência (m³/s)")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/img6.png")"""

    """ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(deseasonalized_inflow, n, T, 2)
    model_prediction = fourier_inflow + ar_prediction

    plt.figure(figsize=(12,8))
    time = np.arange(n)
    plt.plot(time, inflow, ".", linewidth=1, color="black", label="Inflow")
    plt.plot(time, model_prediction, "-", linewidth=1, color="blue", label="Inflow")
    plt.plot(time, fourier_inflow, "--", linewidth=1, color="red", label="Inflow")
    plt.title("Try")
    plt.xlabel("Time")
    plt.ylabel("Inflow")
    plt.legend()
    plt.tight_layout()
    plt.show()

    norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
    ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_ar_predict(norm_deseasonalized_inflow, n, 2)
    model_prediction = fourier_inflow + fourier_std * ar_prediction

    plt.figure(figsize=(12,8))
    time = np.arange(n)
    plt.plot(time, inflow, ".", linewidth=1, color="black", label="Inflow")
    plt.plot(time, model_prediction, "-", linewidth=1, color="blue", label="Inflow")
    plt.plot(time, fourier_inflow, "--", linewidth=1, color="red", label="Inflow")
    plt.title("Try")
    plt.xlabel("Time")
    plt.ylabel("Inflow")
    plt.legend()
    plt.tight_layout()
    plt.show()

    norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
    ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(norm_deseasonalized_inflow, n, T, 2)
    model_prediction = fourier_inflow + fourier_std * ar_prediction

    plt.figure(figsize=(12,8))
    time = np.arange(n)
    plt.plot(time, inflow, ".", linewidth=1, color="black", label="Inflow")
    plt.plot(time, model_prediction, "-", linewidth=1, color="blue", label="Inflow")
    plt.plot(time, fourier_inflow, "--", linewidth=1, color="red", label="Inflow")
    plt.title("Try")
    plt.xlabel("Time")
    plt.ylabel("Inflow")
    plt.legend()
    plt.tight_layout()
    plt.show()"""

main()