import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import kstest, norm


time_divisions = {"sem"}

"""postos = ["14 DE JULHO", "A. VERMELHA", "AIMORES", "ANTA", "APOLONIO SALES", "B. BONITA", "B.COQUEIROS", "BAGUARI", "BAIXO IGUACU", "BALBINA", 
        "BARIRI", "BARRA BRAUNA", "BARRA GRANDE", "BATALHA", "BELO MONTE", "BILL E PEDRAS", "BILLINGS", "BLANG", "BOA ESPERANÃA", "C. DOURADA", 
        "C.BRANCO-1", "C.BRANCO-2", "CACHOEIRA CALDEIRAO", "CACONDE", "CACU", "CAMARGOS", "CAMPOS NOVOS", "CANA BRAVA", "CANAL P. BARRETO", 
        "CANASTRA", "CANDONGA", "CANOAS I", "CANOAS II", "CAPANEMA", "CAPIVARA", "CASTRO ALVES", "CHAVANTES", "COARACY NUNES", "COLIDER", 
        "CORUMBA", "CORUMBA-3", "CORUMBA-4", "CURUA-UNA", "D. FRANCISCA", "DARDANELOS", "DIVISA", "E. DA CUNHA", "EDGARD SOUZA", "EMBORCAÃÃO", 
        "ERNESTINA", "ESPORA", "ESTREITO", "FERREIRA GOMES", "FONTES", "FOZ CHAPECO", "FOZ DO RIO CLARO", "FUNDÃO", "FUNIL", "FUNIL-MG", "FURNAS", 
        "G. B. MUNHOZ", "G. P. SOUZA", "GARIBALDI", "GOV JAYME CANET JR", "GUAPORE", "GUARAPIRANGA", "GUILM. AMORIM", "HENRY BORDEN", "I. SOLTEIRA", 
        "IBITINGA", "IGARAPAVA", "ILHA + T. IRMÃOS", "ILHA POMBOS", "IRAPE", "ITAIPU", "ITAPARICA", "ITAPEBI", "ITAUBA", "ITIQUIRA I", "ITIQUIRA II", 
        "ITUMBIARA", "ITUTINGA", "ITÃ", "JACUI", "JAGUARA", "JAGUARI", "JAURU", "JIRAU", "JORDÃO", "JUPIA", "JURUENA", "JURUMIRIM", "L. C. BARRETO", 
        "LAJEADO", "LAJES", "LIMOEIRO", "LUIZ GONZAGA", "M. MORAES", "MACHADINHO", "MANSO", "MARIMBONDO", "MASCARENHAS", "MAUA", "MIRANDA", "MONJOLINHO", 
        "MONTE CLARO", "MOXOTO", "N. AVANHANDAVA", "NILO PEÃANHA", "NOVA PONTE", "OURINHOS", "P. AFONSO 1,2,3", "P. AFONSO 4", "P. COLOMBIA", "PARAIBUNA", 
        "PARANAPANEMA", "PASSO FUNDO", "PASSO REAL", "PASSO SAO JOAO", "PEDRA DO CAVALO", "PEDRAS", "PEIXE ANGICAL", "PEREIRA PASSOS", "PICADA", "PIMENTAL", 
        "PIRAJU", "PONTE DE PEDRA", "PONTE NOVA", "PORTO ESTRELA", "PORTO PRIMAVERA", "PROMISSÃO", "QUEBRA QUEIXO", "QUEIMADO", "R-11", "RETIRO BAIXO", 
        "RIO BONITO", "RONDON II", "ROSAL", "ROSANA", "S.DO FACÃO", "S.R.VERDINHO", "SA CARVALHO", "SALTO", "SALTO APIACAS", "SALTO CAXIAS", "SALTO GRANDE CM", 
        "SALTO GRANDE CS", "SALTO OSORIO", "SALTO PILAO", "SALTO RS", "SALTO SANTIAGO", "SAMUEL", "SANTA BRANCA", "SANTA CECILIA", "SANTA CLARA-PR", "SANTANA", 
        "SANTO ANTONIO", "SANTONIO CM", "SAO DOMINGOS", "SAO JOSE", "SAO MANOEL", "SAO ROQUE", "SAO SALVADOR", "SEGREDO", "SERRA DA MESA", "SIMPLICIO", "SINOP", 
        "SOBRADINHO", "SOBRADINHO INCR", "SOBRAGI", "STA.CLARA-MG", "STO ANTONIO DO JARI", "SUIÃA", "SÃO SIMÃO", "TAQUARUÃU", "TELES PIRES", "TIBAGI MONTANTE", 
        "TOCOS", "TRÃS IRMÃOS", "TRÃS MARIAS", "TUCURUI", "VIGARIO", "VOLTA GRANDE", "XINGO"]"""

postos = ["SOBRADINHO"]

lenghts = {"dia": 365, "sem": 52, "mes": 12, "est": 4}
harmonics = 3
ar_p = 2
years = 24
number_of_samples = 100


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
    predictions = np.zeros_like(inflow_by_period)
    for _ in range(number_of_samples):
        future_prediction = utils.simple_ar_p_future(deseasonalized_inflow, ar_coef, est_residuals_std, T, years, ar_p)
        future_prediction = future_prediction.reshape((years, T))
        predictions = np.vstack((predictions, future_prediction))
        period_mean += np.mean(future_prediction, axis=0) / T
        period_std += np.std(future_prediction, ddof=1, axis=0) / T
        period_kurtosis += kurtosis(future_prediction, axis=0) / T
        period_ks_stats += [kstest(future_prediction[:, period], inflow_by_period[:, period])[0] / T for period in range(T)]
        period_ks_p += [kstest(future_prediction[:, period], inflow_by_period[:, period])[1] / T for period in range(T)] 

    predictions = predictions[years:]
    residuals_mean = np.mean(residuals)
    residuals_kurtosis = kurtosis(residuals)
    residuals_ks_stats, residuals_ks_p = kstest(residuals, 'norm', args=(0, est_residuals_std))

    utils.sample_vs_normal(residuals, n, residuals_ks_p)
    utils.compare_histogram(predictions[:,0], inflow_by_period[:,0], n, period_ks_p[0])
    utils.compare_histogram(predictions[:,7], inflow_by_period[:,7], n, period_ks_p[7])
    

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
            print(f"EXTRAINDO DADOS - {posto}, {time_division}")
            utils.extract_data(posto)
            for i in range(1,5):
                print(f"RODANDO MODELO {i}")
                try:
                    model_output = pipeline(posto, time_division, i)
                    filename = "model" + str(i) + "_" + time_division + ".csv"
                    utils.save_to_csv(model_output, filename, posto)
                except ValueError: 
                    print("Erro! Entrada contem NaN")


main()