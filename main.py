import utils
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import kurtosis
from scipy.stats import kstest, norm


time_divisions = {"sem", "mes"}
postos = ["SOBRADINHO", "MACHADINHO"]
lenghts = {"dia": 365, "sem": 52, "mes": 12, "est": 4}
harmonics = 3
ar_p = 2
years = 24
number_of_samples = 1


def pipeline(posto, time_division, model):
    inflow, n, T = utils.load_data(posto, time_division)
    fourier_inflow, deseasonalized_inflow, fourier_sq_error, fourier_coef = utils.inflow_fourier_predict(inflow, n, T, harmonics)
    fourier_std, std_fourier_coef = utils.fourier_residuals_std(deseasonalized_inflow, n, T, harmonics)

    if model == 1:
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_ar_predict(deseasonalized_inflow, n, ar_p)
        model_prediction = fourier_inflow + ar_prediction
        
    elif model == 2: 
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(deseasonalized_inflow, n, T, ar_p)
        model_prediction = fourier_inflow + ar_prediction

    elif model == 3: 
        norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_ar_predict(norm_deseasonalized_inflow, n, ar_p)
        model_prediction = fourier_inflow + fourier_std * ar_prediction
    
    else: 
        norm_deseasonalized_inflow = deseasonalized_inflow / fourier_std
        ar_prediction, residuals, ar_coef, ar_sq_error, est_residuals_std = utils.inflow_periodic_ar_predict(norm_deseasonalized_inflow, n, T, ar_p)
        model_prediction = fourier_inflow + fourier_std * ar_prediction

    final_error = np.mean(np.square(inflow - model_prediction))
    nmrse_max = np.sqrt(final_error) / (np.max(inflow) - np.min(inflow))
    nmrse_mean = np.sqrt(final_error) / np.mean(inflow)
    inflow_by_period = inflow.reshape((years, T)) 

    period_mean = np.zeros(T)
    period_std = np.zeros(T)
    period_kurtosis = np.zeros(T)
    period_ks_stats = np.zeros(T)
    period_ks_p = np.zeros(T)
    for _ in range(number_of_samples):
        future_prediction = utils.simple_ar_p_future(deseasonalized_inflow, ar_coef, est_residuals_std, T, years, ar_p)
        future_prediction = future_prediction.reshape((years, T))
        period_mean += np.mean(future_prediction, axis=0) / T
        period_std += np.std(future_prediction, ddof=1, axis=0) / T
        period_kurtosis += kurtosis(future_prediction, axis=0) / T
        period_ks_stats += [kstest(future_prediction[:, period], inflow_by_period[:, period])[0] / T for period in range(T)]
        period_ks_p += [kstest(future_prediction[:, period], inflow_by_period[:, period])[1] / T for period in range(T)] 

    residuals_mean = np.mean(residuals)
    residuals_kurtosis = kurtosis(residuals)
    residuals_ks_stats, residuals_ks_p = kstest(residuals, 'norm', args=(0, est_residuals_std))

    return {"ar_coef": ar_coef, 
            "inflow_fourier_coef": fourier_coef, 
            "std_fourier_coef": std_fourier_coef,
            "sq_error": final_error,
            "normalized_sq_error_mean": nmrse_mean,
            "normalized_sq_error_max": nmrse_max,
            "residuals_mean": residuals_mean,
            "residuals_std": est_residuals_std, 
            "residuals_kurtosis": residuals_kurtosis,
            "residuals_ks_stats": residuals_ks_stats,
            "residuals_ks_p": residuals_ks_p,
            "period_mean": period_mean,
            "period_std": period_std,
            "period_kurtosis": period_kurtosis,
            "period_ks_stats": period_ks_stats,
            "period_ks_p": period_ks_p}


def main():
    for time_division in time_divisions:
        for posto in postos:
            print("EXTRAINDO DADOS")
            utils.extract_data(posto)
            for i in range(4):
                print(f"RODANDO MODELO {i+1}")
                model_output = pipeline(posto, time_division, i)
                filename = "model" + str(i+1) + "_" + time_division + ".csv"
                utils.save_to_csv(model_output, filename, posto)
            
main()