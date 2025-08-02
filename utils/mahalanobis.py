import numpy as np

def mahalanobis(data, Nr, Ptrain):
    """
    Mahalanobis with one COV matrix per class.

    INPUTS:
        data (numpy.ndarray): dataset matrix (N x (p+1)), last column = class labels
        Nr (int): Number of runs (Nr>=1)
        Ptrain (float): Percentage of training data (0 < Ptrain < 100)

    OUTPUTS:
        STATS (list): Statistics of test data (mean, min, max, median, std)
        TX_OK (list): Recognition rates for each run
        X (dict): Data samples separated per class
        m (dict): Class centroids
        S (dict): Covariance matrices per class
        posto (dict): Rank of each covariance matrix
    """
    N, p = data.shape
    Ntrn = int(round(Ptrain * N / 100))
    Ntst = N - Ntrn
    K = int(np.max(data[:, -1]))
    print(f'The problem has {K} classes')

    TX_OK = []
    X = {}
    m = {}
    S = {}
    posto = {}

    for r in range(Nr):
        I = np.random.permutation(N)
        data_shuffled = data[I, :]

        Dtrn = data_shuffled[:Ntrn, :]
        Dtst = data_shuffled[Ntrn:, :]

        # Partition of training data into K subsets
        for k in range(1, K+1):
            idx = np.where(Dtrn[:, -1] == k)[0]
            X[k] = Dtrn[idx, :-1]
            m[k] = np.mean(X[k], axis=0).reshape(-1, 1)
            S[k] = np.cov(X[k], rowvar=False)
            posto[k] = np.linalg.matrix_rank(S[k])
            try:
                iS = np.linalg.inv(S[k])
            except np.linalg.LinAlgError:
                iS = np.linalg.pinv(S[k])
            S[k] = S[k]  # keep original for det, but use iS for distance
            S[f'i{k}'] = iS

        correct = 0
        for i in range(Ntst):
            Xtst = Dtst[i, :-1].reshape(-1, 1)
            Label_Xtst = int(Dtst[i, -1])
            dist = []
            for k in range(1, K+1):
                v = Xtst - m[k]
                iS_k = S[f'i{k}']
                try:
                    det_Sk = np.linalg.det(S[k])
                    if det_Sk <= 0:
                        det_Sk = 1e-10
                except np.linalg.LinAlgError:
                    det_Sk = 1e-10
                d = float(v.T @ iS_k @ v) + np.log(det_Sk)
                dist.append(d)
            Pred_class = np.argmin(dist) + 1  # +1 for 1-based class labels
            if Pred_class == Label_Xtst:
                correct += 1
        TX_OK.append(100 * correct / Ntst)

    STATS = [np.mean(TX_OK), np.min(TX_OK), np.max(TX_OK), np.median(TX_OK), np.std(TX_OK)]
    return STATS, TX_OK, X, m, S, posto