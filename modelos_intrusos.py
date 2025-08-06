import numpy as np
import pandas as pd
from IPython.display import display
from utils.compara_todos import _train_test_split 

def modelo1(X_treino, y_treino, X_teste, intruso, limiar):
    # --- ETAPA DE TREINO ---
    # Filtra apenas as classes de usuários autorizados para criar os modelos
    classes_autorizadas = np.unique(y_treino[y_treino != intruso])
    
    means = {c: np.mean(X_treino[y_treino == c], axis=0) for c in classes_autorizadas}
    covs = {c: np.cov(X_treino[y_treino == c], rowvar=False, bias=True) for c in classes_autorizadas}

    inv_covs, log_dets, thresholds = {}, {}, {}
    for c in classes_autorizadas:
        cov = covs[c]
        try:
            inv_covs[c] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_covs[c] = np.linalg.pinv(cov)

        sign, log_det = np.linalg.slogdet(cov)
        log_dets[c] = log_det if sign > 0 else -np.inf
        
        # --- CÁLCULO DO LIMIAR ---
        # Calcula a distância das amostras de treino para o seu próprio modelo
        X_c = X_treino[y_treino == c]
        diff_c = X_c - means[c]
        distancias_treino = np.sum((diff_c @ inv_covs[c]) * diff_c, axis=1)
        scores_treino = distancias_treino + log_dets[c]
        # Define o limiar como o percentil 95 (aceita 95% dos rostos corretos)
        thresholds[c] = np.percentile(scores_treino, limiar)

    # --- ETAPA DE PREDIÇÃO ---
    # Calcula os scores de cada amostra de teste para cada modelo autorizado
    scores = np.zeros((X_teste.shape[0], len(classes_autorizadas)))
    for i, c in enumerate(classes_autorizadas):
        diff = X_teste - means[c]
        mahalanobis = np.sum((diff @ inv_covs[c]) * diff, axis=1)
        scores[:, i] = mahalanobis + log_dets[c]

    # Encontra a classe autorizada mais próxima e a distância correspondente
    indices_mais_proximos = np.argmin(scores, axis=1)
    distancias_minimas = np.min(scores, axis=1)
    classes_mais_proximas = classes_autorizadas[indices_mais_proximos]
    
    # --- VERIFICAÇÃO COM O LIMIAR ---
    y_pred_final = np.zeros_like(classes_mais_proximas)
    for i in range(len(X_teste)):
        classe_candidata = classes_mais_proximas[i]
        distancia = distancias_minimas[i]
        limiar = thresholds[classe_candidata]
        
        if distancia <= limiar:
            y_pred_final[i] = classe_candidata # Acesso concedido
        else:
            y_pred_final[i] = intruso # Acesso negado, é um intruso
            
    return y_pred_final


def modelo2(X_treino, y_treino, X_teste, intruso, limiar):
    # --- ETAPA DE TREINO ---
    classes_autorizadas = np.unique(y_treino[y_treino != intruso])
    centroides = {c: np.mean(X_treino[y_treino == c], axis=0) for c in classes_autorizadas}
    thresholds = {}
    
    # --- CÁLCULO DO LIMIAR ---
    for c in classes_autorizadas:
        X_c = X_treino[y_treino == c]
        distancias_treino = np.sum((X_c - centroides[c])**2, axis=1)
        thresholds[c] = np.percentile(distancias_treino, limiar)

    # --- ETAPA DE PREDIÇÃO ---
    distancias = np.array([np.sum((X_teste - m)**2, axis=1) for m in centroides.values()]).T
    
    indices_mais_proximos = np.argmin(distancias, axis=1)
    distancias_minimas = np.min(distancias, axis=1)
    classes_mais_proximas = classes_autorizadas[indices_mais_proximos]

    # --- VERIFICAÇÃO COM O LIMIAR ---
    y_pred_final = np.zeros_like(classes_mais_proximas)
    for i in range(len(X_teste)):
        classe_candidata = classes_mais_proximas[i]
        distancia = distancias_minimas[i]
        limiar = thresholds[classe_candidata]
        
        if distancia <= limiar:
            y_pred_final[i] = classe_candidata
        else:
            y_pred_final[i] = intruso
            
    return y_pred_final

def calcular_metricas_binarias(y_true, y_pred, intruso):
    y_true_bin = np.where(y_true == intruso, 1, 0)
    y_pred_bin = np.where(y_pred == intruso, 1, 0)
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0))
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    total = len(y_true_bin)
    acuracia = (tp + tn) / total if total > 0 else 0
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    taxa_fp = fp / (fp + tn) if (fp + tn) > 0 else 0
    taxa_fn = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {'acuracia (%)': acuracia*100, 
            'taxa_falsos_positivos (%)': taxa_fp*100,
            'taxa_falsos_negativos (%)': taxa_fn*100, 
            'sensibilidade (%)': sensibilidade*100,
            'precisao (%)': precisao*100}


def avaliar(modelo, nome_arquivo, Nr, Ptrain, intruso, limiar):
    """Executa a avaliação binária para o modelo."""
    D = np.loadtxt(nome_arquivo)
    X, y = D[:, :-1], D[:, -1].astype(int)

    lista_de_metricas = []
    for i in range(Nr):
        X_treino, X_teste, y_treino, y_teste = _train_test_split(X, y, Ptrain / 100.0, i)
        
        # --- PASSAR O RÓTULO DO INTRUSO PARA A FUNÇÃO ---
        y_pred = modelo(X_treino, y_treino, X_teste, intruso, limiar)

        metricas_rodada = calcular_metricas_binarias(y_teste, y_pred, intruso)
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