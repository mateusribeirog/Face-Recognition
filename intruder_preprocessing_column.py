import os
import cv2
import numpy as np
import re 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from face_preprocessing_column import normalize_zscore


def processar_imagens(dimensao_img, 
                      aplicar_pca,  
                      tipo_normalizacao,
                      nome_arquivo_saida):
    """
    Carrega, processa e vetoriza imagens de faces de forma flexível.
    """
    print("--- Fase 1: Carregando e Processando Imagens ---")
    base_path = 'data_intruso'
    if not os.path.isdir(base_path):
        print(f"ERRO: A pasta '{base_path}' não foi encontrada.")
        return

    X_list, Y_list = [], []

    todos_os_arquivos = sorted(os.listdir(base_path))

    print(f"Encontrados {len(todos_os_arquivos)} arquivos na pasta '{base_path}'. Processando...")

    for nome_arquivo in todos_os_arquivos:
        match = re.match(r'subject(\d+)\..*', nome_arquivo)
        
        if not match:
            continue
            
        rotulo = int(match.group(1))
        
        file_path = os.path.join(base_path, nome_arquivo)
        
        Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if Img is None:
            print(f"  Aviso: Falha ao ler o arquivo '{file_path}'. Pulando.")
            continue

        Ar = cv2.resize(Img, dimensao_img)
        A = Ar.astype(np.float64) / 255.0
        a = A.flatten('F')
        
        X_list.append(a)
        Y_list.append(rotulo)

    if not X_list:
        print("\nERRO CRÍTICO: Nenhuma imagem válida foi carregada.")
        return

    X = np.column_stack(X_list)
    Y = np.array(Y_list)
    print(f"\nCarregamento concluído. {len(np.unique(Y))} classes encontradas.")
    
    if aplicar_pca:
        print(f"\n--- Aplicando PCA ---")
        pca = PCA(n_components=None) # Pega todos os componentes primeiro
        pca.fit(X.T)

        # % [V L VEi]=pcacov(cov(X'));
        V = pca.components_.T             # Autovetores (componentes principais)
        VEi = pca.explained_variance_ratio_ # Variância explicada por cada componente
        VEq = np.cumsum(VEi)
        variance = 0.98
        q = np.searchsorted(VEq, variance) + 1
        print(f"O valor de q encontrado foi: {q}")
        Vq = V[:, :q]
        Qq = Vq.T
        X_pca = Qq @ X # Projeta os dados originais nos novos componentes
        print(f"Shape da matriz X após projeção no PCA: {X_pca.shape}")
        X = X_pca
    if tipo_normalizacao != 'nenhuma':
        print(f"\n--- Aplicando normalização: {tipo_normalizacao} ---")
        if tipo_normalizacao == 'z-score': 
            X = normalize_zscore(X)
        

    print(f"\n--- Salvando dados processados em '{nome_arquivo_saida}' ---")
    Z = np.vstack([X, Y]).T
    np.savetxt(nome_arquivo_saida, Z, fmt='%.8f')
    print("Arquivo salvo com sucesso.")


if __name__ == '__main__':
    print('--- Gerando para o primeiro modelo ---')
    processar_imagens((20, 20), False, 'nenhuma', 'modelo_1_data.dat')
    print('--- Gerando para o segundo modelo ---')    
    processar_imagens((20, 20), True, 'z-score', 'modelo_2_data.dat')
    