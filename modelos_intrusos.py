import numpy as np
from utils.compara_todos import _train_test_split
import pandas as pd
from IPython.display import display

def modelo1(X_treino, y_treino, X_teste):
    classes = np.unique(y_treino)
    means = {c: np.mean(X_treino[y_treino == c], axis = 0) for c in classes}
    covs = {c: np.cov(X_treino[y_treino == c], rowvar=False, bias = True) for c in classes}

    inv_covs, log_dets = {}, {}

    for c in classes:
        cov = covs[c]
        
        try:
            inv_covs[c] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_covs[c] = np.linalg.pinv(cov)

        sign, log_det = np.linalg.slogdet(cov)
        log_dets[c] = log_det if sign > 0 else -np.inf


    scores = np.zeros((X_teste.shape[0], len(classes)))
    for i, c in enumerate(classes):
        diff = X_teste - means[c]
        mahalanobis = np.sum((diff @ inv_covs[c]) * diff, axis = 1)
        scores[:,i] = mahalanobis + log_dets[c]

    y_pred = classes[np.argmin(scores, axis=1)]

    return y_pred


def modelo2(X_treino, y_treino, X_teste):
    classes = np.unique(y_treino)
    centroides = {c: np.mean(X_treino[y_treino == c], axis=0) for c in classes}
    distancias = np.array([np.sum((X_teste - m)**2, axis=1) for m in centroides.values()]).T
    y_pred = classes[np.argmin(distancias, axis=1)]
    return y_pred

def calcular_metricas_binarias(y_true, y_pred, intruso):
    """
    Calcula as métricas.
    '1' é o intruso, '0' é um não-intruso.
    """
    # Converte os rótulos multi-classe para binário
    y_true_bin = np.where(y_true == intruso, 1, 0)
    y_pred_bin = np.where(y_pred == intruso, 1, 0)

    # Calcula Verdadeiros/Falsos Positivos/Negativos
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1)) # Intruso previsto corretamente
    tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0)) # Não-intruso previsto corretamente
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0)) # Não-intruso classificado como intruso
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1)) # Intruso classificado como não-intruso

    # Cálculo das métricas
    total = len(y_true_bin)
    acuracia = (tp + tn) / total if total > 0 else 0
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    taxa_fp = fp / (fp + tn) if (fp + tn) > 0 else 0
    taxa_fn = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'acuracia (%)': acuracia*100,
        'taxa_falsos_positivos (%)': taxa_fp*100,
        'taxa_falsos_negativos (%)': taxa_fn*100,
        'sensibilidade (%)': sensibilidade*100,
        'precisao (%)': precisao*100
    }

def avaliar(modelo, nome_arquivo, Nr, Ptrain, intruso):
    """
    Executaca a avaliação binária para o modelo
    """

    D = np.loadtxt(nome_arquivo)
    X, y = D[:, :-1], D[:, -1]

    lista_de_metricas = []
    for i in range(Nr):
        X_treino, X_teste, y_treino, y_teste = _train_test_split(X, y, Ptrain/100, i)

        y_pred_multiclasse = modelo(X_treino, y_treino, X_teste)

        metricas_rodada = calcular_metricas_binarias(y_teste, y_pred_multiclasse, intruso)
        lista_de_metricas.append(metricas_rodada)
    
    df_runs = pd.DataFrame(lista_de_metricas)

    stats = {
        'Média': df_runs.mean(),
        'Mínimo':df_runs.min(),
        'Maximo':df_runs.max(),
        'Mediana': df_runs.median(),
        'Desvio Padrão': df_runs.std(),
    }
    pd.options.display.float_format = '{:,.2f}'.format
    return pd.DataFrame(stats)

