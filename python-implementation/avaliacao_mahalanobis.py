import numpy as np
import matplotlib.pyplot as plt
from mahalanobis import mahalanobis
import time

# Load data (assuming whitespace-delimited)
D = np.loadtxt('coluna2.dat')

Nr = 100  # Number of repetitions
Ptrain = 80  # Percentage of training

start_time = time.time()
STATS_0, TX_OK0, X0, m0, S0, posto0 = mahalanobis(D, Nr, Ptrain)
Tempo0 = time.time() - start_time

print("STATS_0:", STATS_0)
print("Tempo0:", Tempo0)

plt.figure()
plt.boxplot(TX_OK0)
plt.title('Boxplot of Recognition Rates')
plt.xlabel('Runs')
plt.ylabel('Recognition Rate (%)')

plt.figure()
plt.hist(TX_OK0, bins=15, edgecolor='black', alpha=0.7)
plt.title('Histogram of Recognition Rates')
plt.xlabel('Recognition Rate (%)')
plt.ylabel('Frequency')
plt.show()