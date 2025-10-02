import numpy as np
import pandas as pd
import utils
import time
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.special import gamma

years = 24 # quantidade de anos no dataset (2001 - 2024)


def get_cv_indices(n_total, time_division, fold_index):
    years_per_fold = 6
    if time_division == "dia": samples_per_year = 365
    elif time_division == "sem": samples_per_year = 52
    else: samples_per_year = 12
    
    samples_in_fold = int(samples_per_year * years_per_fold)
    test_start = fold_index * samples_in_fold
    test_end = min(test_start + samples_in_fold, n_total)
    
    all_indices = np.arange(n_total)
    test_indices = all_indices[test_start:test_end]
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    return train_indices, test_indices

def calculate_aic(n, sse, k):
    if sse <= 0 or n <= 0: return np.inf
    return n * np.log(sse / n) + 2 * k

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def run_grid_search_analysis(postos, time_divisions, p_range, h_range, h_std_range):
    results = []
    
    for posto in postos:
        for time_division in time_divisions:
            full_inflow, n, T = utils.load_data(posto, time_division)
            
            for model_id in range(1, 5):
                current_h_std_range = h_std_range if model_id in [3, 4] else [None]

                for p in p_range:
                    for h in h_range:
                        for h_std in current_h_std_range:
                            start_time = time.time()
                            
                            # --- VALIDAÇÃO CRUZADA ---
                            fold_errors = []
                            for fold in range(4):
                                train_idx, test_idx = get_cv_indices(n, time_division, fold)
                                inflow_train, inflow_test = full_inflow[train_idx], full_inflow[test_idx]
                                n_train = len(inflow_train)

                                # Fourier ajustado apenas no treino
                                f_results_train = utils.inflow_fourier_predict(inflow_train, n_train, T, h)
                                f_coef_train = f_results_train["coef"]
                                d_inflow_train = f_results_train["residuals"]  # residuals of train set

                                # Previsão Fourier para o teste: usar matriz com índices absolutos do tempo
                                fourier_matrix_test = utils.create_fourier_matrix(test_idx, T, h)
                                fourier_pred_test = fourier_matrix_test @ f_coef_train

                                # Para dessazonalizar full series: aplicar coef. de treino a todos os índices
                                full_fourier_matrix = utils.create_fourier_matrix(np.arange(n), T, h)
                                full_fourier_pred_from_train = full_fourier_matrix @ f_coef_train
                                # NOTA: esta dessazonalização aplica os coef. estimados no treino a toda série; é aceitável
                                full_deseasonalized = full_inflow - full_fourier_pred_from_train

                                # Agora ajustar AR somente no train (usar os mesmos pontos relativos)
                                if model_id == 1:
                                    # AR simples ajustado em treino (deseasonalized[train_idx])
                                    ar_pred_full, _, _, _, _ = utils.inflow_ar_predict(full_deseasonalized[train_idx], n_train, p)
                                    # ar_pred_full is predicted for positions 0..n_train-1 relative to train block:
                                    # we need predictions aligned to test absolute indices. -> create full-length pred and map
                                    # assume utils.inflow_ar_predict returns array length n_train (predictions aligned to input)
                                    # We'll build a vector of length n with NaNs and fill train positions
                                    ar_pred_vector = np.full(n, np.nan)
                                    ar_pred_vector[train_idx] = ar_pred_full
                                    model_prediction_test = fourier_pred_test + ar_pred_vector[test_idx]

                                elif model_id == 2:
                                    ar_pred_full, ar_coeffs, _, _, _ = utils.inflow_periodic_ar_predict(full_deseasonalized[train_idx], T, p)
                                    ar_pred_vector = np.full(n, np.nan)
                                    ar_pred_vector[train_idx] = ar_pred_full
                                    model_prediction_test = fourier_pred_test + ar_pred_vector[test_idx]

                                else:
                                    # modelos 3 e 4: std estimado a partir do train residuals
                                    _, std_f_coef_train = utils.fourier_residuals_std(d_inflow_train, n_train, T, h_std)
                                    # construir std em toda a série usando coef estimados no treino
                                    full_fourier_std_matrix = utils.create_fourier_matrix(np.arange(n), T, h_std)
                                    full_fourier_std = full_fourier_std_matrix @ std_f_coef_train
                                    full_fourier_std[full_fourier_std < 1e-6] = 1e-6

                                    # normalizar apenas no treino para ajustar o AR
                                    norm_train = (full_deseasonalized[train_idx] / full_fourier_std[train_idx])

                                    if model_id == 3:
                                        ar_pred_norm_full, _, _, _, _ = utils.inflow_ar_predict(norm_train, len(norm_train), p)
                                        ar_pred_norm_vector = np.full(n, np.nan)
                                        ar_pred_norm_vector[train_idx] = ar_pred_norm_full
                                        # reconstruir na escala original para testar
                                        ar_pred_reconst = full_fourier_std * ar_pred_norm_vector
                                        model_prediction_test = fourier_pred_test + ar_pred_reconst[test_idx]

                                    else:
                                        ar_pred_norm_full, ar_coeffs, _, _, _ = utils.inflow_periodic_ar_predict(norm_train, T, p)
                                        ar_pred_norm_vector = np.full(n, np.nan)
                                        ar_pred_norm_vector[train_idx] = ar_pred_norm_full
                                        ar_pred_reconst = full_fourier_std * ar_pred_norm_vector
                                        model_prediction_test = fourier_pred_test + ar_pred_reconst[test_idx]

                                # calcular MSE no conjunto de teste (na escala original)
                                mse = calculate_mse(inflow_test, model_prediction_test)
                                fold_errors.append(mse)

                            avg_cv_mse = np.mean(fold_errors)

                            # --- 2. TREINO FINAL E AIC (SEM DATA-LEAKAGE) ---
                            fourier_results = utils.inflow_fourier_predict(full_inflow, n, T, h)
                            d_inflow = fourier_results["residuals"]
                            k_ar = p   # valor padrão (caso simples, AR normal)

                            if model_id == 1:
                                ar_pred_full, _, _, ar_sq_error_mean, _ = utils.inflow_ar_predict(d_inflow, n, p)
                                sse = ar_sq_error_mean * n

                            elif model_id == 2:
                                ar_pred_full, ar_coeffs, _, ar_sq_error_mean, _ = utils.inflow_periodic_ar_predict(d_inflow, T, p)
                                sse = ar_sq_error_mean * n
                                if ar_coeffs is not None:
                                    k_ar = len(ar_coeffs)

                            elif model_id == 3:
                                full_fourier_std, _ = utils.fourier_residuals_std(d_inflow, n, T, h_std)
                                full_fourier_std[full_fourier_std < 1e-6] = 1e-6
                                norm_d_inflow = d_inflow / full_fourier_std
                                ar_pred_norm_full, _, _, ar_sq_error_mean, _ = utils.inflow_ar_predict(norm_d_inflow, n, p)
                                ar_pred_reconst = full_fourier_std * ar_pred_norm_full
                                resid = d_inflow - ar_pred_reconst
                                sse = np.sum(resid**2)

                            else:  # model_id == 4
                                full_fourier_std, _ = utils.fourier_residuals_std(d_inflow, n, T, h_std)
                                full_fourier_std[full_fourier_std < 1e-6] = 1e-6
                                norm_d_inflow = d_inflow / full_fourier_std
                                ar_pred_norm_full, ar_coeffs, _, ar_sq_error_mean, _ = utils.inflow_periodic_ar_predict(norm_d_inflow, T, p)
                                ar_pred_reconst = full_fourier_std * ar_pred_norm_full
                                resid = d_inflow - ar_pred_reconst
                                sse = np.sum(resid**2)
                                if ar_coeffs is not None:
                                    k_ar = len(ar_coeffs)

                            # contabilizar parâmetros
                            k_fourier = (2 * h + 1)
                            k_h_std = (2 * h_std + 1) if h_std is not None else 0
                            total_k = k_fourier + k_h_std + k_ar
                            total_aic = calculate_aic(n, sse, total_k)

                            # --- 3. ARMAZENAR RESULTADOS --- 
                            results.append({'posto': posto,
                                            'time_division': time_division,
                                            'model_id': model_id, 
                                            'p': p,
                                            'h': h,
                                            'h_std': h_std if h_std is not None else 0,
                                            'avg_cv_mse': avg_cv_mse,
                                            'total_aic': total_aic}) 
                            end_time = time.time()
                print(model_id, time_division, posto)
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    P_RANGE = range(1, 6)
    H_RANGE = range(1, 6)
    H_STD_RANGE = range(1, 6)
    POSTOS_EXEC = ["ITAIPU", "MACHADINHO", "I. SOLTEIRA", "TUCURUI", "ITUMBIARA"]
    TIME_DIVISIONS_EXEC = ["dia", "sem", "mes"]

    print("Iniciando a análise de grid search")
    
    final_results_df = run_grid_search_analysis(
        postos=POSTOS_EXEC,
        time_divisions=TIME_DIVISIONS_EXEC,
        p_range=P_RANGE,
        h_range=H_RANGE,
        h_std_range=H_STD_RANGE
    )

    print("\n\n========================= RESULTADOS FINAIS DO GRID SEARCH =========================")
    print(final_results_df.to_string())

    output_filename = "grid_search_model_results.csv"
    final_results_df.to_csv(output_filename, index=False)
    print(f"\nResultados completos salvos em '{output_filename}'")