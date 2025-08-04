import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import display

from utils.modelos import (
    dmc_classifier,
    nn1_classifier,
    maxcorr_classifier,
    qda_classifier
)

def _accuracy_score(y_true, y_pred):
    """Calcula a taxa de acerto"""
    return np.sum(y_true == y_pred) / len(y_true)

def _train_test_split(X, y, p_train, random_state):
    """
    Divide os dados em treino e teste usando uma permutação.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices_permutados = np.random.permutation(n_samples)
    ponto_divisao = int(n_samples * p_train)
    indices_treino = indices_permutados[:ponto_divisao]
    indices_teste = indices_permutados[ponto_divisao:]
    X_treino, X_teste = X[indices_treino], X[indices_teste]
    y_treino, y_teste = y[indices_treino], y[indices_teste]
    
    return X_treino, X_teste, y_treino, y_teste


def compara_todos(nome_arquivo: str, Nr: int, Ptrain: float):
    """Executa a análise comparativa de múltiplos classificadores."""
    try:
        print(f"--- CARREGANDO DADOS DE '{nome_arquivo}' ---")
        D = np.loadtxt(nome_arquivo)
        X, y = D[:, :-1], D[:, -1]
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo '{nome_arquivo}' não encontrado.")
        return None, None

    # Dicionário mapeando nomes a funções (usando lambda para passar argumentos extras)
    classifier_funcs = {
        "Quadrático": lambda tr_x, tr_y, te_x: qda_classifier(tr_x, tr_y, te_x, mode='QDA_Puro'),
        "Variante 1(Tikhonov)": lambda tr_x, tr_y, te_x: qda_classifier(tr_x, tr_y, te_x, mode='Tikhonov', regularization_param=0.01),
        "Variante 2(Pooled)": lambda tr_x, tr_y, te_x: qda_classifier(tr_x, tr_y, te_x, mode='LDA'),
        "Variante 3(Friedman)": lambda tr_x, tr_y, te_x: qda_classifier(tr_x, tr_y, te_x, mode='Friedman', regularization_param=0.5),
        "Variante 4(Naive Bayes)": lambda tr_x, tr_y, te_x: qda_classifier(tr_x, tr_y, te_x, mode='NaiveBayes'),
        "MaxCorr": maxcorr_classifier,
        "DMC": dmc_classifier,
        "1-NN": nn1_classifier
    }

    results = []
    all_tx_ok = {}
    all_last_run_info = {}
    
    print(f"\n--- INICIANDO AVALIAÇÃO ({Nr} repetições, {Ptrain}% treino) ---")

    for name, func in classifier_funcs.items():
        print(f"Avaliando: {name}...")
        start_time = time.time()
        tx_ok_list = []
        total_failed_inversions = 0

        for i in range(Nr):
            X_treino, X_teste, y_treino, y_teste = _train_test_split(X, y, Ptrain / 100.0, random_state=i)
            
            y_pred, model_params, failed_count = func(X_treino, y_treino, X_teste)
            
            total_failed_inversions += failed_count
            acc = _accuracy_score(y_teste, y_pred)
            tx_ok_list.append(acc * 100)

            if i == Nr - 1:
                model_params['X'] = {c: X_treino[y_treino == c] for c in np.unique(y_treino)}
                all_last_run_info[name] = model_params
        
        execution_time = time.time() - start_time
        
        all_tx_ok[name] = tx_ok_list
        tx_ok_array = np.array(tx_ok_list)
        stats = [
            np.mean(tx_ok_array), np.min(tx_ok_array), np.max(tx_ok_array),
            np.median(tx_ok_array), np.std(tx_ok_array)
        ]
        results.append(stats + [total_failed_inversions, execution_time])
            
    # Montagem do DataFrame
    df_results = pd.DataFrame(
        results,
        columns=['Média', 'Mínimo', 'Máximo', 'Mediana', 'Desvio Padrão', 'Matrizes Singulares', 'Tempo (s)'],
        index=classifier_funcs.keys()
    )
    
    pd.options.display.float_format = '{:,.2f}'.format
    display(df_results)
    
    return df_results, all_last_run_info
