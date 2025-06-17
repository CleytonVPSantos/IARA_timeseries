import utils
import matplotlib.pyplot as plt
import numpy as np

postos = ["SOBRADINHO"]
lenghts = {"dia": 365, "sem": 52, "mes": 12, "est": 4}
harmonics = 3
ar_p = 2

def pipeline1(posto):
    for period in ["sem"]:
        inflow, n, T = utils.load_data(posto, "sem")
        fourier_inflow, deseasonalized_inflow, fourier_sq_error, fourier_coef = utils.inflow_fourier_predict(inflow, n, T, harmonics)
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_ar_predict(deseasonalized_inflow, n, ar_p)
        model_prediction = fourier_inflow + ar_prediction
        final_error = np.mean(np.square(inflow - model_prediction))
        print(final_error)
        time = np.arange(n)
        plt.figure(figsize=(12, 4))
        plt.plot(time, inflow, '.', color='red', markersize=4, label='dados reais')
        plt.plot(time, model_prediction, '-', color='blue', markersize=4, label='dados reais')

        plt.title(f"Previsao do modelo")
        plt.xlabel("Período")
        plt.ylabel("Vazão (m³/s)")
        plt.grid(True, linestyle=':', linewidth=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model_prediction 

def pipeline2(posto):
    for period in ["sem"]:
        inflow, n, T = utils.load_data(posto, "sem")
        fourier_inflow, deseasonalized_inflow, fourier_sq_error, fourier_coef = utils.inflow_fourier_predict(inflow, n, T, harmonics)
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(deseasonalized_inflow, n, T, ar_p)
        model_prediction = fourier_inflow + ar_prediction
        final_error = np.mean(np.square(inflow - model_prediction))
        print(final_error)
        time = np.arange(n)
        plt.figure(figsize=(12, 4))
        plt.plot(time, inflow, '.', color='red', markersize=4, label='dados reais')
        plt.plot(time, model_prediction, '-', color='blue', markersize=4, label='dados reais')

        plt.title(f"Previsao do modelo")
        plt.xlabel("Período")
        plt.ylabel("Vazão (m³/s)")
        plt.grid(True, linestyle=':', linewidth=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model_prediction 


def pipeline3(posto):
    for period in ["sem"]:
        inflow, n, T = utils.load_data(posto, "sem")
        fourier_inflow, deseasonalized_inflow, fourier_sq_error, fourier_coef = utils.inflow_fourier_predict(inflow, n, T, harmonics)
        fourier_std = utils.fourier_residuals_std(deseasonalized_inflow, n, T, 5)
        norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_ar_predict(norm_deseasonalized_inflow, n, ar_p)
        model_prediction = fourier_inflow + fourier_std * ar_prediction
        final_error = np.mean(np.square(inflow - model_prediction))
        print(final_error)
        time = np.arange(n)
        plt.figure(figsize=(12, 4))
        plt.plot(time, inflow, '.', color='red', markersize=4, label='dados reais')
        plt.plot(time, model_prediction, '-', color='blue', markersize=4, label='dados reais')

        plt.title(f"Previsao do modelo")
        plt.xlabel("Período")
        plt.ylabel("Vazão (m³/s)")
        plt.grid(True, linestyle=':', linewidth=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model_prediction 


def pipeline4(posto):
    for period in ["sem"]:
        inflow, n, T = utils.load_data(posto, "sem")
        fourier_inflow, deseasonalized_inflow, fourier_sq_error, fourier_coef = utils.inflow_fourier_predict(inflow, n, T, harmonics)
        fourier_std = utils.fourier_residuals_std(deseasonalized_inflow, n, T, 5)
        norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(norm_deseasonalized_inflow, n, T, ar_p)
        model_prediction = fourier_inflow + fourier_std * ar_prediction
        final_error = np.mean(np.square(inflow - model_prediction))
        print(final_error)
        time = np.arange(n)
        plt.figure(figsize=(12, 4))
        plt.plot(time, residuals, '-', color='blue', markersize=4, label='dados reais')

        plt.title(f"Previsao do modelo")
        plt.xlabel("Período")
        plt.ylabel("Vazão (m³/s)")
        plt.grid(True, linestyle=':', linewidth=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()


        
    return model_prediction


def main():
    for posto in postos:
        utils.extract_data(posto)
        model1_prediction = pipeline1(posto)
        model2_prediction = pipeline2(posto)
        model3_prediction = pipeline3(posto)
        model4_prediction = pipeline4(posto)

main()