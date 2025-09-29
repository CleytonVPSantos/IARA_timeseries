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

# lê os arquivos com os dados originais, trata os dados e escreve em um csv separando por posto e tamanho do periodo
def extract_data(reservoir):
    years = range(2001, 2025)
    filenames = [f"data/DADOS_HIDROLOGICOS_RES_{year}.csv" for year in years]

    dh = []
    for filename in filenames:
        df = pd.read_csv(filename, sep=';')

        # converter a coluna de data
        df['din_instante'] = pd.to_datetime(df['din_instante'], errors='coerce')

        # remover 29/02
        df = df[~((df['din_instante'].dt.month == 2) & (df['din_instante'].dt.day == 29))]

        dh.append(df)

    # filtrar o reservatório
    data = [df[df['nom_reservatorio'] == reservoir] for df in dh]

    periods = [1, 7, 30, 90, 180]
    group_by = ['dia', 'sem', 'mes', 'est', 'met']

    path = "./"
    dir_list = os.listdir(path) 
    names = ['data/hydro/' + reservoir + '_' + T + '.csv' for T in group_by]

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

            # junta todos os anos em um só arquivo csv
            pd.concat(data_by_year).to_csv(names[i], sep=';', index=False)


# carrega os dados a partir do nome do reservatório e unidade de tempo
def load_data(reservoir, group_by):
    inflow = pd.read_csv('data/hydro/' + reservoir + "_" + group_by + ".csv", sep=';')["val_vazaoincremental"].to_numpy()
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


# returns the seasonal component estimation using fourier series
def inflow_fourier_predict(inflow, n, T, N):
    model = LinearRegression(fit_intercept=False)
    time = np.arange(n)
    
    fourier_matrix = create_fourier_matrix(time, T, N)
    model.fit(fourier_matrix, inflow)
    inflow_fourier_pred = model.predict(fourier_matrix)
    residuals = inflow - inflow_fourier_pred
    coef = model.coef_
    # Estimativa de sigma^2
    sigma2 = np.sum(residuals**2) / n
    # Log-verossimilhança
    ll = -0.5 * n * (np.log(2*np.pi*sigma2) + 1)
    # AIC
    AIC = 2*(2*N + 1) - 2*ll
    return inflow_fourier_pred, residuals, np.square(residuals).mean, coef, AIC


# calculate data sample variance and mean for each period separately
def mean_and_std(data, periods):
    data = data.reshape((years, periods))
    data_mean = np.tile(np.mean(data, axis=0), years)
    data_std = np.tile(np.std(data, axis=0, ddof=1), years)
    return data_mean, data_std


# ajusta fourier no desvio padrão dos residuos
def fourier_residuals_std(deseasonalized_inflow, n, T, N):
    residuals_std = mean_and_std(deseasonalized_inflow, T)[1]
    residuals_fourier_pred, _, _, std_fourier_coef, _  = inflow_fourier_predict(residuals_std, n, T, N)
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

    model = LinearRegression(fit_intercept=False)
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

    model = LinearRegression(fit_intercept=False)
    
    model.fit(ar_regression_matrix, residuals_to_pred)
    residuals_ar_predict = np.hstack((residuals[:p], model.predict(ar_regression_matrix)))
    final_residuals = residuals - residuals_ar_predict

    residuals_std_b = np.std(final_residuals)
    sq_error = np.square(final_residuals).mean()
    
    return residuals_ar_predict, final_residuals, model.coef_, sq_error, residuals_std_b


# Simula um AR(p) por um determinado número de periodos, dados parâmetros e dados iniciais
def simple_ar_p_future(data, phi, sigma, T, additional_years, p):
    current_data = list(data[-p:])
    future_residuals = []
    total_forecast_steps = additional_years * T

    phi_arr = np.array(phi)

    for _ in range(total_forecast_steps):
        next_val = np.dot(phi_arr, current_data[::-1]) + sigma * np.random.normal()
        
        future_residuals.append(next_val)

        current_data.pop(0)
        current_data.append(next_val)

    return np.array(future_residuals)


# Simula um AR(p) com coeficientes fixos por um determinado número de periodos, dados parâmetros e dados iniciais
def periodic_ar_p_future(data, phi, sigma, T, additional_years, p, last_observed_period_idx):
    current_data = list(data[-p:])
    future_residuals = []
    total_forecast_steps = additional_years * T
    
    phi_matrix = np.array(phi).reshape(T, p)

    for i in range(total_forecast_steps):
        current_period_index = (last_observed_period_idx + 1 + i) % T
        
        current_phi = phi_matrix[current_period_index]

        next_val = np.dot(current_phi, current_data[::-1]) + sigma * np.random.normal()
        
        future_residuals.append(next_val)

        current_data.pop(0)
        current_data.append(next_val)

    return np.array(future_residuals)


# salva os dados de uma simulação em um csv
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


def plot_hist(data, n, filename):
    """Salva um histograma exibindo a média e o desvio padrão na legenda."""
    # Cálculo de bins pela regra de Freedman-Diaconis
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Prevenção de divisão por zero
    if iqr > 0:
        h = 2 * iqr * n **(-1/3)
        range_val = np.max(data) - np.min(data)
        opt_bin = int(range_val / h) if h > 0 else 20
    else:
        opt_bin = 20
    
    # Calcular média e desvio padrão
    mu = np.mean(data)
    sigma = np.std(data, ddof=1) # ddof=1 para desvio padrão amostral

    # Formatar a string para a legenda
    label_text = f'Amostra\n($\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$)'

    plt.figure(figsize=(8, 4))
    # Usar o novo texto na legenda
    plt.hist(data, bins=opt_bin, color='lightblue', edgecolor='black', label=label_text)
    plt.xlabel("Inflow (m³/s)")
    plt.ylabel("Frequência")
    plt.title("Histograma dos Dados")
    plt.grid(True, linestyle=':', linewidth=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def sample_vs_normal(data, n, filename):
    """Salva um histograma vs. PDF Normal, exibindo média e desvio padrão."""
    # Cálculo de bins
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    if iqr > 0:
        h = 2 * iqr * n **(-1/3)
        range_val = np.max(data) - np.min(data)
        opt_bin = int(range_val / h) if h > 0 else 20
    else:
        opt_bin = 20

    # Média e desvio padrão já são calculados
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    # Geração da PDF Normal
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)
    stat, p_value = kstest(data, 'norm', args=(mu, sigma))
    
    # Formatar a string para a legenda do histograma
    hist_label = f'Amostra\n($\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$)'

    plt.figure(figsize=(8, 4))
    # Usar o novo texto na legenda do histograma
    plt.hist(data, bins=opt_bin, density=True, color='lightblue', edgecolor='black', label=hist_label)
    plt.plot(x, pdf, 'r-', lw=2, label='PDF Normal Teórica')
    plt.xlabel("Vazão (m³/s)")
    plt.ylabel("Densidade")
    plt.title(f"Amostra vs Normal (p-valor = {p_value:.4f})")
    plt.grid(True, linestyle=':', linewidth=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_histogram_vertical(data1, data2, filename):
    """Salva histogramas comparativos, cada um com sua média e desvio padrão."""
    # Cálculos gerais para bins e eixos
    combined_data = np.concatenate([data1, data2])
    min_val = np.min(combined_data)
    max_val = np.max(combined_data)
    stat, p_value = kstest(data1, data2)
    n_total = len(combined_data)
    q1_total = np.percentile(combined_data, 25)
    q3_total = np.percentile(combined_data, 75)
    iqr_total = q3_total - q1_total

    if iqr_total > 0:
        bin_width = 2 * iqr_total * n_total ** (-1/3)
        num_bins = int((max_val - min_val) / bin_width)
    else:
        num_bins = int(1 + np.log2(n_total))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Comparativo de Distribuições (p-valor K-S = {p_value:.4f})", fontsize=16)

    mu1 = np.mean(data1)
    sigma1 = np.std(data1, ddof=1)
    label1 = f'Dados Reais\n($\\mu={mu1:.2f}$, $\\sigma={sigma1:.2f}$)'
    
    ax1.hist(data1, bins=num_bins, range=(min_val, max_val),
             color='#007ACC', edgecolor='black', label=label1)
    ax1.set_ylabel("Frequência")
    ax1.set_title("Dados Reais")
    ax1.grid(True, linestyle=':', linewidth=0.5)
    ax1.legend()

    mu2 = np.mean(data2)
    sigma2 = np.std(data2, ddof=1)
    label2 = f'Simulação\n($\\mu={mu2:.2f}$, $\\sigma={sigma2:.2f}$)'
    
    ax2.hist(data2, bins=num_bins, range=(min_val, max_val),
             color='#FF4E50', edgecolor='black', label=label2)
    ax2.set_xlabel("Vazão (m³/s)")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Simulação")
    ax2.grid(True, linestyle=':', linewidth=0.5)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    

def plot(data_set, n, markers, colors, labels, title, xlable, ylable, filename):
    plt.figure(figsize=(12,8))
    time = np.arange(n)
    for i, data in enumerate(data_set):
        plt.plot(time, data, markers[i], linewidth=1, color=colors[i], label=labels[i])
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


res = ["14 DE JULHO", "A. VERMELHA", "AIMORES", "ANTA", "APOLONIO SALES", "B. BONITA", "B.COQUEIROS", "BAGUARI", 
          "BAIXO IGUACU", "BALBINA", "BARIRI", "BARRA BRAUNA", "BARRA GRANDE", "BATALHA", "BELO MONTE", "BILL E PEDRAS", 
          "BILLINGS", "BLANG", "BOA ESPERANÇA", "C. DOURADA", "C.BRANCO-1", "C.BRANCO-2", "CACHOEIRA CALDEIRAO", "CACONDE", 
          "CACU", "CAMARGOS", "CAMPOS NOVOS", "CANA BRAVA", "CANAL P. BARRETO", "CANASTRA", "CANDONGA", "CANOAS I", 
          "CANOAS II", "CAPANEMA", "CAPIVARA", "CASTRO ALVES", "CHAVANTES", "COARACY NUNES", "COLIDER", "CORUMBA", "CORUMBA-3", 
          "CORUMBA-4", "CURUA-UNA", "D. FRANCISCA", "DARDANELOS", "DIVISA", "E. DA CUNHA", "EDGARD SOUZA", "EMBORCAÇÃO", 
          "ERNESTINA", "ESPORA", "ESTREITO", "FERREIRA GOMES", "FONTES", "FOZ CHAPECO", "FOZ DO RIO CLARO", "FUNDÃO", "FUNIL", 
          "FUNIL-MG", "FURNAS", "G. B. MUNHOZ", "G. P. SOUZA", "GARIBALDI", "GOV JAYME CANET JR", "GUAPORE", "GUARAPIRANGA", 
          "GUILM. AMORIM", "HENRY BORDEN", "I. SOLTEIRA", "IBITINGA", "IGARAPAVA", "ILHA + T. IRMÃOS", "ILHA POMBOS", "IRAPE", 
          "ITAIPU", "ITAPARICA", "ITAPEBI", "ITAUBA", "ITIQUIRA I", "ITIQUIRA II", "ITUMBIARA", "ITUTINGA", "ITÁ", "JACUI", 
          "JAGUARA", "JAGUARI", "JAURU", "JIRAU", "JORDÃO", "JUPIA", "JURUENA", "JURUMIRIM", "L. C. BARRETO", "LAJEADO", "LAJES", 
          "LIMOEIRO", "LUIZ GONZAGA", "M. MORAES", "MACHADINHO", "MANSO", "MARIMBONDO", "MASCARENHAS", "MAUA", "MIRANDA", "MONJOLINHO", 
          "MONTE CLARO", "MOXOTO", "N. AVANHANDAVA", "NILO PEÇANHA", "NOVA PONTE", "OURINHOS", "P. AFONSO 1,2,3", "P. AFONSO 4", 
          "P. COLOMBIA", "PARAIBUNA", "PARANAPANEMA", "PASSO FUNDO", "PASSO REAL", "PASSO SAO JOAO", "PEDRA DO CAVALO", "PEDRAS", 
          "PEIXE ANGICAL", "PEREIRA PASSOS", "PICADA", "PIMENTAL", "PIRAJU", "PONTE DE PEDRA", "PONTE NOVA", "PORTO ESTRELA", 
          "PORTO PRIMAVERA", "PROMISSÃO", "QUEBRA QUEIXO", "QUEIMADO", "R-11", "RETIRO BAIXO", "RIO BONITO", "RONDON II", "ROSAL", 
          "ROSANA", "S.DO FACÃO", "S.R.VERDINHO", "SA CARVALHO", "SALTO", "SALTO APIACAS", "SALTO CAXIAS", "SALTO GRANDE CM", 
          "SALTO GRANDE CS", "SALTO OSORIO", "SALTO PILAO", "SALTO RS", "SALTO SANTIAGO", "SAMUEL", "SANTA BRANCA", "SANTA CECILIA", 
          "SANTA CLARA-PR", "SANTANA", "SANTO ANTONIO", "SANTONIO CM", "SAO DOMINGOS", "SAO JOSE", "SAO MANOEL", "SAO ROQUE", "SAO SALVADOR", 
          "SEGREDO", "SERRA DA MESA", "SIMPLICIO", "SINOP", "SOBRADINHO", "SOBRADINHO INCR", "SOBRAGI", "STA.CLARA-MG", "STO ANTONIO DO JARI", 
          "SUIÇA", "SÃO SIMÃO", "TAQUARUÇU", "TELES PIRES", "TIBAGI MONTANTE", "TOCOS", "TRÊS IRMÃOS", "TRÊS MARIAS", "TUCURUI", "VIGARIO", "VOLTA GRANDE", "XINGO"]
