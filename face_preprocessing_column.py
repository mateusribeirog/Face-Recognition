import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# % Routines for opening face images and convert them to column vectors
# % by stacking the columns of the face matrix one beneath the other.
# %
# % Last modification: 10/08/2021
# % Author: Guilherme Barreto

print("Iniciando pré-processamento de imagens...")

def normalize_zscore(X):
    """Aplica normalização Z-score (média 0, desvio padrão 1)."""
    # Transpõe para (amostras, features) para calcular média/std por feature
    X_T = X.T
    mean = np.mean(X_T, axis=0)
    std = np.std(X_T, axis=0)
    # Adiciona um valor pequeno (epsilon) para evitar divisão por zero
    X_T_norm = (X_T - mean) / (std + 1e-8)
    # Transpõe de volta para (features, amostras)
    return X_T_norm.T

def normalize_minmax_01(X):
    """Aplica normalização Min-Max para o intervalo [0, 1]."""
    X_T = X.T
    min_val = np.min(X_T, axis=0)
    max_val = np.max(X_T, axis=0)
    X_T_norm = (X_T - min_val) / (max_val - min_val + 1e-8)
    return X_T_norm.T

def normalize_minmax_11(X):
    """Aplica normalização Min-Max para o intervalo [-1, 1]."""
    # Reutiliza a normalização [0, 1] e depois a converte para [-1, 1]
    X_norm_01 = normalize_minmax_01(X)
    X_norm_11 = 2 * X_norm_01 - 1
    return X_norm_11

def box_cox(X):
    X_pos = X + 1e-6
    X_transformed = np.zeros_like(X_pos)
    for i in range(X_pos.shape[0]):
        X_transformed[i, :], _ = boxcox(X_pos[i, :])
    print("Transformação Box-Cox concluída.")
    return X_transformed

def pre_process(norm, pca, Q, output_filename):

    #####################################################
    # % Fase 1 -- Carrega imagens disponiveis
    #####################################################
    part1 = 'subject0'
    part2 = 'subject'
    part3 = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad', '.sleepy', '.surprised', '.wink']

    Nind = 15          # % Quantidade de individuos (classes)
    Nexp = len(part3)  # % Quantidade de expressoes

    X_list = []  # % Lista para acumular imagens vetorizadas
    Y_list = []  # % Lista para acumular o rotulo (identificador) do individuo

    # --- Loop principal para carregar e processar as imagens ---
    for i in range(1, Nind + 1):  # % Indice para os individuos
        print(f"Processando indivíduo: {i}")  # % individuo=i,
        for j in range(Nexp):  # % Indice para expressoes
            
            # % Monta o nome do arquivo de imagem
            if i < 10:
                nome_base = f"{part1}{i}{part3[j]}"
            else:
                nome_base = f"{part2}{i}{part3[j]}"

            # Lógica para encontrar o arquivo de imagem na pasta 'data'
            file_path = None
            # Procura por arquivos com várias extensões (incluindo sem extensão)
            for ext in ['', '.gif', '.pgm', '.jpg', '.png', '.jpeg']:
                potential_path = os.path.join('data', nome_base + ext)
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
            
            if not file_path:
                print(f"  Aviso: Imagem para '{nome_base}' não encontrada. Pulando.")
                continue

            # % le imagem (em escala de cinza)
            Img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if Img is None:
                print(f"  Aviso: Falha ao ler o arquivo de imagem '{file_path}'. Pulando.")
                continue

            # % (Opcional) Redimensiona imagem
            Ar = cv2.resize(Img, (20, 20))

            # % (Opcional) adiciona ruido (mantido como no original, sem ruído)
            An = Ar
            
            # % converte (im2double) para double precision no intervalo [0, 1]
            A = An.astype(np.float64) / 255.0
            
            # % Etapa de vetorizacao: Empilhamento das colunas
            a = A.flatten('F')
            
            # % Rotulo = indice do individuo
            ROT = i
            
            # Adiciona os vetores às listas
            X_list.append(a)
            Y_list.append(ROT)

    # Converte as listas para matrizes NumPy
    # % Coloca cada imagem vetorizada como coluna da matriz X
    X = np.column_stack(X_list) 
    # % Coloca o rotulo de cada vetor como coluna da matriz Y
    Y = np.array(Y_list)

    print(f"\nMatriz de dados X criada com shape: {X.shape}")
    print(f"Vetor de rótulos Y criado com shape: {Y.shape}")

    if norm == 'z-score': X = normalize_zscore(X)
    elif norm == 'minmax01': X = normalize_minmax_01(X)
    elif norm == 'minmax11': X = normalize_minmax_11(X)
    elif norm == 'boxcox': X = box_cox(X)

    if pca:
        print("\nAplicando PCA...")

        # Scikit-learn espera dados com amostras por linha, então transpomos X
        pca = PCA(n_components=None) # Pega todos os componentes primeiro
        pca.fit(X.T)

        # % [V L VEi]=pcacov(cov(X'));
        V = pca.components_.T             # Autovetores (componentes principais)
        VEi = pca.explained_variance_ratio_ # Variância explicada por cada componente
        VEq = np.cumsum(VEi)
        variance = 0.98
        if Q == 400:
            q = 400
            print(f"\nAplicado pca com q = 400")

        else:
            q = np.searchsorted(VEq, variance) + 1
            # % q=25; Vq=V(:,1:q); Qq=Vq'; X=Qq*X;
            print(f"\nO valor de q encontrado foi: {q}")

        Vq = V[:, :q]
        Qq = Vq.T
        X = Qq @ X # Projeta os dados originais nos novos componentes
        
        print(f"Shape da matriz X após projeção no PCA: {X.shape}")

        if norm == 'boxcox':
            plt.figure(figsize=(8, 6))
            plt.plot(VEq, 'r-', linewidth=3)
            plt.title('Variância Explicada Acumulada pelo PCA com Box Cox')
            plt.axvline(x = q, color = 'blue', linestyle = ':', label = f'{q} componentes')
            plt.axhline(y = variance, color = 'green', linestyle = ':', label = f'{variance*100}% da variância')
            plt.xlabel('Número de Componentes Principais') # % xlabel('Autovalor'); (Mais preciso em Python)
            plt.ylabel('Variância Explicada Acumulada') # % ylabel('Variancia explicada acumulada');
            plt.grid(True)
            plt.legend()
            plt.savefig('veq_pcaboxcox.png')
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            plt.plot(VEq, 'r-', linewidth=3)
            plt.title('Variância Explicada Acumulada pelo PCA')
            plt.axvline(x = q, color = 'blue', linestyle = ':', label = f'{q} componentes')
            plt.axhline(y = variance, color = 'green', linestyle = ':', label = f'{variance*100}% da variância')
            plt.xlabel('Número de Componentes Principais') # % xlabel('Autovalor'); (Mais preciso em Python)
            plt.ylabel('Variância Explicada Acumulada') # % ylabel('Variancia explicada acumulada');
            plt.grid(True)
            plt.legend()
            plt.savefig('veq_pca.png')
            plt.show()

    Z = np.vstack([X, Y])

    # % Z=Z'; 
    # Transpõe para que cada linha seja uma amostra
    Z = Z.T

    np.savetxt(output_filename, Z, fmt='%.8f')




if __name__ == '__main__':
    print("\n>>> Gerando: Dados para Atividade 1 (Sem PCA, Sem Normalização)")
    pre_process(norm='nenhuma', pca=False, Q=None, 
                output_filename=f'recfaces_20x20_sem_pca_sem_norm.dat')

    print("\n>>> Gerando: Dados para OBS-2 (Sem PCA, Z-Score)")
    pre_process(norm='z-score', pca=False, Q=None, 
                output_filename=f'recfaces_20x20_sem_pca_zscore.dat')

    print("\n>>> Gerando: Dados para OBS-2 (Sem PCA, Min-Max [0, 1])")
    pre_process(norm='minmax01', pca=False, Q=None, 
                output_filename=f'recfaces_20x20_sem_pca_minmax01.dat')

    print("\n>>> Gerando: Dados para OBS-2 (Sem PCA, Min-Max [-1, 1])")
    pre_process(norm='minmax11', pca=False, Q=None, 
                output_filename=f'recfaces_20x20_sem_pca_minmax11.dat')

    print("\n>>> Gerando: Dados para Atividade 3 (PCA sem redução)")
    pre_process(norm='nenhuma', pca=True, Q=400, 
                output_filename=f'recfaces_20x20_pca_sem_reducao.dat')

    print("\n>>> Gerando: Dados para Atividade 5 (PCA com redução)")
    pre_process(norm='nenhuma', pca=True, Q=None, 
                output_filename=f'recfaces_20x20_pca_com_reducao.dat')

    print("\n>>> Gerando: Dados para Atividade 7 (Box-Cox + PCA com redução)")
    pre_process(norm='boxcox', pca=True, Q=None, 
                output_filename=f'recfaces_20x20_boxcox_pca.dat')

    print("\n--- Processo finalizado! Todos os 7 arquivos .dat foram gerados. ---")