import utils
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import shapiro, normaltest

def cv_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_y = np.mean(y_true)
    return rmse / mean_y * 100 if mean_y != 0 else np.nan

def save_table_as_image(df, output_path="resultados_recent.png"):    
    # Formata números com 2 casas decimais
    df_to_display = df.copy()
    for col in df_to_display.select_dtypes(include=[float, int]).columns:
        df_to_display[col] = df_to_display[col].map(lambda x: f"{x:.2f}")
    
    n_rows, n_cols = df_to_display.shape
    
    fig_width = max(12, n_cols * 1.2)
    fig_height = max(2, n_rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.axis('off')

    table = ax.table(cellText=df_to_display.values,
                     colLabels=df_to_display.columns,
                     cellLoc='center',
                     loc='center')
  
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(n_cols)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tabela final salva como imagem em: {output_path}")

def run_tests(inflow, model_prediction, i, T, posto, time_division, fourier, std_residuals, ar_coef, ar_p):
    residuo_final = inflow - model_prediction
    n = len(residuo_final)
    
    #Ljung–Box 
    lags = 20
    lb_test = acorr_ljungbox(residuo_final, lags=[lags], return_df=True)
    lb_stat = lb_test["lb_stat"].iloc[0]
    lb_pvalue = lb_test["lb_pvalue"].iloc[0]
    
    # Shapiro–Wilk 
    sample = residuo_final[:min(n, 5000)]
    shapiro_stat, shapiro_p = shapiro(sample)
    
    #ARCH (heterocedasticidade) 
    arch_stat, arch_p, _, _ = het_arch(residuo_final)
    
    #Teste de simetria (D’Agostino)
    dagostino_stat, dagostino_p = normaltest(sample)
    
    # Métricas de ajuste 
    ss_res = np.sum(residuo_final ** 2)
    ss_tot = np.sum((inflow - np.mean(inflow)) ** 2)
    r2 = 1 - ss_res / ss_tot
    k = len(ar_coef) if hasattr(ar_coef, "__len__") else 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    mae = np.mean(np.abs(residuo_final))
    kurt = pd.Series(residuo_final).kurtosis()
    skew = pd.Series(residuo_final).skew()
    resid_mean = np.mean(residuo_final)
    resid_std = np.std(residuo_final)
    
    # CV(RMSE)
    cv_rmse_value = cv_rmse(inflow, model_prediction)
    
    # Monta resultado
    result = {
        "Modelo": i,
        "AR_p": ar_p,
        "Posto": posto,
        "TimeDivision": time_division,
        "Lags_LB": lags,
        "LjungBox_stat": lb_stat,
        "LjungBox_p": lb_pvalue,
        "Shapiro_stat": shapiro_stat,
        "Shapiro_p": shapiro_p,
        "ARCH_stat": arch_stat,
        "ARCH_p": arch_p,
        "DAgostino_stat": dagostino_stat,
        "DAgostino_p": dagostino_p,
        "MAE": mae,
        "R2": r2,
        "R2_adj": r2_adj,
        "Resid_Mean": resid_mean,
        "Resid_Std": resid_std,
        "Resid_Skew": skew,
        "Resid_Kurt": kurt,
        "CV_RMSE": cv_rmse_value 
    }
    return result

def main():
    harmonics = 4
    harmonics_2 = 10

    for ar_p in [1, 2, 5]:
        for posto in ["MACHADINHO", "ITAIPU", "I. SOLTEIRA", "TUCURUI"]:
            for time_division in ["dia", "sem", "mes"]:
                inflow, n, T = utils.load_data(posto, time_division)
                fourier = utils.inflow_fourier_predict(inflow, n, T, harmonics)
                fourier_std, std_fourier_coef = utils.fourier_residuals_std(fourier["residuals"], n, T, harmonics_2)

                all_results = []

                for model in range(1, 5):
                    if model == 1:
                        ar_prediction, residuals, ar_coef, _, _ = utils.inflow_ar_predict(fourier["residuals"], n, ar_p)
                        model_prediction = fourier["pred"] + ar_prediction

                    elif model == 2:
                        ar_prediction, residuals, ar_coef, _, _ = utils.inflow_periodic_ar_predict(fourier["residuals"], T, ar_p)
                        model_prediction = fourier["pred"] + ar_prediction

                    elif model == 3:
                        norm_res = fourier["residuals"] / fourier_std
                        ar_prediction, residuals, ar_coef, _, _ = utils.inflow_ar_predict(norm_res, n, ar_p)
                        model_prediction = fourier["pred"] + fourier_std * ar_prediction

                    else:
                        norm_res = fourier["residuals"] / fourier_std
                        ar_prediction, residuals, ar_coef, _, _ = utils.inflow_periodic_ar_predict(norm_res, T, ar_p)
                        model_prediction = fourier["pred"] + fourier_std * ar_prediction

                    # Salva resultado em lista
                    result = run_tests(inflow, model_prediction, model, T,
                                            posto, time_division, fourier, fourier_std, ar_coef, ar_p)
                    all_results.append(result)

                # cria tabela com os 4 modelos e salva imagem 
                df_results = pd.DataFrame(all_results)
                display_cols = ["Modelo", "LjungBox_p", "Shapiro_p", "ARCH_p", 
                                "DAgostino_p", "MAE", "R2", "R2_adj", "Resid_Kurt", "CV_RMSE"]
                display_df = df_results[display_cols]

                output_path = f"{posto}_AR{ar_p}_{time_division}_resultados.png"
                save_table_as_image(display_df, output_path)

    # Salva resultados em CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("diagnostic_tests_results.csv", index=False)
    print(df_results.head())

main()
