# avalia_todos.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import display

# Importa as funções de classificação do arquivo modelos.py
from modelos import (
    dmc_classifier,
    nn1_classifier,
    maxcorr_classifier,
    qda_puro_classifier,
    qda_regularizado_classifier
)

# --- Funções Utilitárias ---
def _custom_accuracy_score(y_true, y_pred):
    """Calcula a acurácia (taxa de acerto)."""
    return np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0

def _custom_train_test_split(X, y, p_train, random_state):
    """Divide os dados em treino e teste de forma estratificada."""
    np.random.seed(random_state)
    classes, y_indices = np.unique(y, return_inverse=True)
    train_idx, test_idx = [], []
    for c in range(classes.shape[0]):
        class_indices = np.where(y_indices == c)[0]
        np.random.shuffle(class_indices)
        n_train_c = int(len(class_indices) * p_train)
        train_idx.extend(class_indices[:n_train_c])
        test_idx.extend(class_indices[n_train_c:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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
        "QDA (Puro)": qda_puro_classifier,
        "Variante 1 (Tikhonov)": lambda tr_x, tr_y, te_x: qda_regularizado_classifier(tr_x, tr_y, te_x, mode='Tikhonov', regularization_param=0.01),
        "Variante 2 (Pooled)": lambda tr_x, tr_y, te_x: qda_regularizado_classifier(tr_x, tr_y, te_x, mode='LDA'),
        "Variante 3 (Friedman)": lambda tr_x, tr_y, te_x: qda_regularizado_classifier(tr_x, tr_y, te_x, mode='Friedman', regularization_param=0.5),
        "Variante 4 (Naive Bayes)": lambda tr_x, tr_y, te_x: qda_regularizado_classifier(tr_x, tr_y, te_x, mode='NaiveBayes'),
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
            X_treino, X_teste, y_treino, y_teste = _custom_train_test_split(X, y, Ptrain / 100.0, random_state=i)
            
            y_pred, model_params, failed_count = func(X_treino, y_treino, X_teste)
            
            total_failed_inversions += failed_count
            acc = _custom_accuracy_score(y_teste, y_pred)
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
        
    print("\n--- AVALIAÇÃO FINALIZADA ---\n")
    
    # Montagem do DataFrame
    df_results = pd.DataFrame(
        results,
        columns=['Média (%)', 'Mínimo (%)', 'Máximo (%)', 'Mediana (%)', 'Desvio Padrão', 'Matrizes Singulares (Total)', 'Tempo (s)'],
        index=classifier_funcs.keys()
    )
    
    pd.options.display.float_format = '{:,.2f}'.format
    print("--- TABELA DE RESULTADOS ---")
    display(df_results)
    
    # Plotagem do Boxplot
    all_tx_ok_df = pd.DataFrame(all_tx_ok)
    plt.style.use('default')
    plt.figure(figsize=(14, 8))
    all_tx_ok_df.boxplot(patch_artist=True, grid=False)
    
    plt.title(f'Comparação de Desempenho (Dataset: {nome_arquivo})', fontsize=16, pad=20)
    plt.ylabel('Taxa de Acerto (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return df_results, all_last_run_info


# --- Bloco Principal de Execução ---
if __name__ == '__main__':

    np.loadtxt('recfaces.dat')
        
    # Chama a função principal de avaliação
    df_res, info = compara_todos(
        nome_arquivo='recfaces.dat',
        Nr=50,
        Ptrain=80.0
    )