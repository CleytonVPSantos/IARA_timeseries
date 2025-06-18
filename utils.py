# imports necessários
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from scipy.special import gamma
from scipy.stats import kstest, norm
from statsmodels.regression.linear_model import yule_walker


years = 24 # quantidade de anos no dataset (2001 - 2024)


def extract_data(reservoir):
    filenames = [
                "DADOS_HIDROLOGICOS_RES_2001",
                "DADOS_HIDROLOGICOS_RES_2002",
                "DADOS_HIDROLOGICOS_RES_2003",
                "DADOS_HIDROLOGICOS_RES_2004",
                "DADOS_HIDROLOGICOS_RES_2005",
                "DADOS_HIDROLOGICOS_RES_2006",
                "DADOS_HIDROLOGICOS_RES_2007",
                "DADOS_HIDROLOGICOS_RES_2008",
                "DADOS_HIDROLOGICOS_RES_2009",
                "DADOS_HIDROLOGICOS_RES_2010",
                "DADOS_HIDROLOGICOS_RES_2011",
                "DADOS_HIDROLOGICOS_RES_2012",
                "DADOS_HIDROLOGICOS_RES_2013",
                "DADOS_HIDROLOGICOS_RES_2014",
                "DADOS_HIDROLOGICOS_RES_2015",
                "DADOS_HIDROLOGICOS_RES_2016",
                "DADOS_HIDROLOGICOS_RES_2017",
                "DADOS_HIDROLOGICOS_RES_2018",
                "DADOS_HIDROLOGICOS_RES_2019",
                "DADOS_HIDROLOGICOS_RES_2020",
                "DADOS_HIDROLOGICOS_RES_2021",
                "DADOS_HIDROLOGICOS_RES_2022",
                "DADOS_HIDROLOGICOS_RES_2023",
                "DADOS_HIDROLOGICOS_RES_2024"
                ]

    dh = [pd.read_csv(f"data/{filename}.csv", sep=';') for filename in filenames]

    # Filtrar o reservatório
    data = [df[df['nom_reservatorio'] == reservoir] for df in dh]

    periods = [1, 7, 30, 90, 180]
    group_by = ['dia', 'sem', 'mes', 'est', 'met']

    path = "./"
    dir_list = os.listdir(path) 
    names = ['data/hidro/' + reservoir + '_' + T + '.csv' for T in group_by]

    for i in range(len(periods)):
        data_by_year = []

        if names[i] not in dir_list:
            for j in range(len(filenames)):
                df = data[j].copy()
                p = periods[i]

                # garante que o índice está contínuo
                df = df.reset_index(drop=True)

                # remove linhas excedentes para garantir blocos completos
                n_rows = df.shape[0]
                n_full_groups = n_rows // p
                df_clean = df.iloc[:n_full_groups * p].copy()

                # reindexa para o agrupamento funcionar
                df_clean = df_clean.reset_index(drop=True)

                # agrupa por blocos de 'p' linhas
                group_ids = df_clean.index // p

                df_numeric = df_clean.select_dtypes(include='number')
                df_grouped = df_numeric.groupby(group_ids).mean()

                df_non_numeric = df_clean.select_dtypes(exclude='number').groupby(group_ids).first()

                # junta numéricos e não numéricos
                df_result = pd.concat([df_non_numeric, df_grouped], axis=1)

                data_by_year.append(df_result)

            # junta os 24 dfs em um só arquivo csv
            pd.concat(data_by_year).to_csv(names[i], sep=';', index=False)


# carrega os dados a partir do nome do reservatório e unidade de tempo
def load_data(reservoir, group_by):
    inflow = pd.read_csv('data/hidro/' + reservoir + "_" + group_by + ".csv", sep=';')["val_vazaoincremental"].to_numpy()
    n = len(inflow)
    T = int(n / years)

    return inflow, n, T


# cria matriz de fourier para ajuste com minimos quadrados
def create_fourier_matrix(t, T, harmonics):
    features = [np.ones_like(t)]
    for i in range(1, harmonics + 1):
        features.append(np.sin(2*np.pi*i*t/T))
        features.append(np.cos(2*np.pi*i*t/T))

    return np.column_stack(features)


def inflow_fourier_predict(inflow, n, T, N):
    model = LinearRegression()
    time = np.arange(n)
    
    fourier_matrix = create_fourier_matrix(time, T, N)
    model.fit(fourier_matrix, inflow)
    inflow_fourier_pred = model.predict(fourier_matrix)
    residuals = inflow - inflow_fourier_pred
    print(model.intercept_)
    return inflow_fourier_pred, residuals, np.square(residuals).mean, model.coef_


# calculate data sample variance and mean for each period separately
def mean_and_std(data, periods):
    data = data.reshape((years, periods))
    data_mean = np.tile(np.mean(data, axis=0), years)
    data_std = np.tile(np.std(data, axis=0, ddof=1), years)
    return data_mean, data_std


# ajusta fourier no desvio padrão dos residuos
def fourier_residuals_std(deseasonalized_inflow, n, T, N):
    residuals_std = mean_and_std(deseasonalized_inflow, T)[1]
    residuals_fourier_pred, _, _, std_fourier_coef  = inflow_fourier_predict(residuals_std, n, T, N)
    return residuals_fourier_pred, std_fourier_coef


# retorna residuos padronizados e o vetor com os desvios padrao
def padronize_residuals(residuals):
    T = int(len(residuals)/years)
    residuals = residuals.reshape((years, T))
    residuals_std_b = np.std(residuals, axis=0, ddof=1)
    correct = gamma(T/2) / (np.sqrt(T/2) * gamma((T-1)/2))
    residuals_std = residuals_std_b * correct
    residuals_std = np.tile(residuals_std, 24)
    residuals = residuals.flatten()
    residuals = residuals / residuals_std

    return residuals, residuals_std


def least_squares_ar_fit(residuals, n, p):
    predict = residuals[p:]
    predictors = np.vstack([np.ones_like(predict)] + [residuals[p - i: n - i] for i in range(1, p + 1)]).T

    model = LinearRegression()
    model.fit(predictors, predict)

    return np.hstack((residuals[:p], model.predict(predictors))), model.coef_[1:], model.intercept_


def inflow_ar_predict(residuals, n, p):
    # fit do AR nos residuos
    residual_fit, phi, _ = least_squares_ar_fit(residuals, n, p)
    residuals_std_b = np.std(residual_fit - residuals)

    # previsão final = sazonalizada + AR
    final_residuals = residuals - residual_fit
    return residual_fit, final_residuals, phi, np.square(final_residuals).mean(), residuals_std_b


def create_periodic_ar_matrix(inflow, n, T, p):
    ar_regression_matrix = np.zeros((n-p, T*p))
    for i in range(p, n):
        j = i % T
        ar_regression_matrix[i-p, j*p:(j+1)*p] = np.flip(inflow[(i-p):i])

    return ar_regression_matrix


# Ajuste do AR com coeficientes periódicos
def inflow_periodic_ar_predict(residuals, n, T, p):
    residuals_to_pred = residuals[p:]
    ar_regression_matrix = create_periodic_ar_matrix(residuals, n, T, p)

    model = LinearRegression()
    
    model.fit(ar_regression_matrix, residuals_to_pred)
    residuals_ar_predict = np.hstack((residuals[:p], model.predict(ar_regression_matrix)))
    final_residuals = residuals - residuals_ar_predict

    residuals_std_b = np.std(final_residuals)
    sq_error = np.square(final_residuals).mean()
    
    return residuals_ar_predict, final_residuals, model.coef_, sq_error, residuals_std_b


def simple_ar_p_future(data, phi, sigma, T, additional_years, p):
    current_data = list(data[-p:])
    future_residuals = []

    total_forecast_steps = additional_years * T

    for _ in range(total_forecast_steps):
        next_val = 0.0
        for j in range(p):
            next_val += phi[j] * current_data[p - 1 - j]

        next_val += sigma * np.random.normal() 
        future_residuals.append(next_val)

        current_data.pop(0)
        current_data.append(next_val)

    return np.array(future_residuals)


def periodic_ar_p_future(data, phi, sigma, T, additional_years, p, last_observed_period_idx):
    current_data = list(data[-p:])
    future_residuals = []

    total_forecast_steps = additional_years * T

    for i in range(total_forecast_steps):
        current_period_index = (last_observed_period_idx + 1 + i) % T

        next_val = 0.0
        for j in range(p):
            phi_val = phi[current_period_index * p + j]
            next_val += phi_val * current_data[p - 1 - j]

        current_sigma = sigma[current_period_index]
        next_val += current_sigma * np.random.normal()
        future_residuals.append(next_val)

        current_data.pop(0)
        current_data.append(next_val)

    return np.array(future_residuals)


def save_to_csv(data, filename, posto):
    to_csv = {"posto": posto} 

    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            if key in ["ar_coef", "inflow_fourier_coef", "std_fourier_coef"]:
                for i, coef in enumerate(value):
                    to_csv[f"{key}_{i + 1}"] = coef
            elif key in ["period_mean", "period_std", "period_kurtosis", "period_ks_stats", "period_ks_p"]:
                for i, p_val in enumerate(value):
                    to_csv[f"period_{i + 1}_{'_'.join(key.split('_')[1:])}"] = p_val
            else:
                to_csv[key] = str(value)
        else:
            to_csv[key] = value

    df = pd.DataFrame([to_csv])

    try:
        file_exists = os.path.exists(filename)
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
        print(f"Dados salvos com sucesso em '{filename}'")
    except Exception as e:
        print(f"Erro ao salvar os dados no arquivo CSV: {e}")


      
def sample_vs_normal(data, n, p_value):
    # IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    h = 2 * iqr * n **(-1/3)

    # tamanho otimo do bin
    range = np.max(data) - np.min(data)
    opt_bin = int(range // h)

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=opt_bin, density=True, color='lightblue', edgecolor='black', label='Histograma da amostra')
    plt.plot(x, pdf, 'r-', lw=2, label='PDF Normal')
    plt.xlabel("Vazão (m³/s)")
    plt.ylabel("Densidade")
    plt.title(f"Amostra vs Normal (mesma média e desvio padrão) (p valor = {p_value})")
    plt.grid(True, linestyle=':', linewidth=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    stat, p_value = kstest(data, 'norm', args=(mu, sigma))
    print(f'Estatística (KS): {stat:.4f}, p-valor: {p_value:.4f}')


def compare_histogram(data1, data2, n, p_value):        
    # IQR
    q1 = np.percentile(data1, 25)
    q3 = np.percentile(data1, 75)
    iqr = q3 - q1
    h = 2 * iqr * n **(-1/3)

    # tamanho otimo do bin
    range = np.max(data1) - np.min(data1)
    opt_bin1 = int(range // h)

    # IQR
    q1 = np.percentile(data2, 25)
    q3 = np.percentile(data2, 75)
    iqr = q3 - q1
    h = 2 * iqr * n **(-1/3)

    # tamanho otimo do bin
    range = np.max(data2) - np.min(data2)
    opt_bin2 = int(range // h)

    plt.hist(data1, bins=opt_bin1, color='blue', edgecolor='black')
    plt.hist(data2, bins=opt_bin2, color='red', edgecolor='black')
    plt.xlabel("Vazão (m³/s)")
    plt.ylabel("Frequência")
    plt.title(f"Distribuiçoes (p valor = {p_value})")
    plt.grid(True, linestyle=':', linewidth=0.25)
    plt.tight_layout()
    plt.show()