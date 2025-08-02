import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# % Routines for opening face images and convert them to column vectors
# % by stacking the columns of the face matrix one beneath the other.
# %
# % Last modification: 10/08/2021
# % Author: Guilherme Barreto

print("Iniciando pré-processamento de imagens...")

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
        Ar = cv2.resize(Img, (30, 30))

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

"""# %%%%%%%% APLICACAO DE PCA (PCACOV) %%%%%%%%%%%
print("\nAplicando PCA...")

# Scikit-learn espera dados com amostras por linha, então transpomos X
pca = PCA(n_components=None) # Pega todos os componentes primeiro
pca.fit(X.T)

# % [V L VEi]=pcacov(cov(X'));
V = pca.components_.T             # Autovetores (componentes principais)
VEi = pca.explained_variance_ratio_ # Variância explicada por cada componente

# % q=25; Vq=V(:,1:q); Qq=Vq'; X=Qq*X;
q = 25
Vq = V[:, :q]
Qq = Vq.T
X_pca = Qq @ X # Projeta os dados originais nos novos componentes

print(f"Shape da matriz X após projeção no PCA: {X_pca.shape}")

# % VEq=cumsum(VEi); figure; plot(VEq,'r-','linewidth',3);
VEq = np.cumsum(VEi)
plt.figure(figsize=(8, 6))
plt.plot(VEq, 'r-', linewidth=3)
plt.title('Variância Explicada Acumulada pelo PCA')
plt.xlabel('Número de Componentes Principais') # % xlabel('Autovalor'); (Mais preciso em Python)
plt.ylabel('Variância Explicada Acumulada') # % ylabel('Variancia explicada acumulada');
plt.grid(True)
plt.show()
"""
# % Z=[X;Y];  
# Empilha os dados projetados pelo PCA e os rótulos
Z = np.vstack([X, Y])

# % Z=Z'; 
# Transpõe para que cada linha seja uma amostra
Z = Z.T

# % save -ascii recfaces.dat Z
output_filename = 'recfaces.dat'
np.savetxt(output_filename, Z, fmt='%.8f')

print(f"\nProcesso concluído. Arquivo '{output_filename}' salvo com sucesso.")