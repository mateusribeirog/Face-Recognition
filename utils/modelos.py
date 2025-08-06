import numpy as np


def dmc_classifier(X_treino, y_treino, X_teste):
    classes = np.unique(y_treino)
    centroides = {c: np.mean(X_treino[y_treino == c], axis=0) for c in classes}
    distancias = np.array([np.sum((X_teste - m)**2, axis=1) for m in centroides.values()]).T
    y_pred = classes[np.argmin(distancias, axis=1)]
    return y_pred, {'m': centroides, 'S': {}, 'posto': {}}, 0

def nn1_classifier(X_treino, y_treino, X_teste):
    y_pred = np.empty(X_teste.shape[0], dtype=y_treino.dtype)
    for i, x_teste in enumerate(X_teste):
        distancias = np.sum((X_treino - x_teste)**2, axis=1)
        y_pred[i] = y_treino[np.argmin(distancias)]
    return y_pred, {'m': {}, 'S': {}, 'posto': {}}, 0

def maxcorr_classifier(X_treino, y_treino, X_teste):
    classes = np.unique(y_treino)
    norm_centroids = {}
    for c in classes:
        m = np.mean(X_treino[y_treino == c], axis=0)
        norm = np.linalg.norm(m)
        norm_centroids[c] = m / (norm + 1e-9)
    norms_X_teste = np.linalg.norm(X_teste, axis=1, keepdims=True)
    X_teste_norm = X_teste / (norms_X_teste + 1e-9)
    correlacoes = np.array([X_teste_norm @ nc for nc in norm_centroids.values()]).T
    y_pred = classes[np.argmax(correlacoes, axis=1)]
    return y_pred, {'m': norm_centroids, 'S': {}, 'posto': {}}, 0

def qda_classifier(X_treino, y_treino, X_teste, mode='QDA_Puro', regularization_param=0.01):
    """
    Classificador Quadrático e suas variantes.
    """
    classes = np.unique(y_treino)
    n_features = X_treino.shape[1]
    
    priors = {c: (y_treino == c).sum() / len(y_treino) for c in classes}
    means = {c: np.mean(X_treino[y_treino == c], axis=0) for c in classes}
    raw_covs = {c: np.cov(X_treino[y_treino == c], rowvar=False, bias=True) for c in classes}
    
    if mode in ['LDA', 'Friedman']:
        pooled_cov = sum(priors[c] * raw_covs[c] for c in classes)

    final_covs, inv_covs, log_dets = {}, {}, {}
    failed_inversions_count = 0

    for c in classes:
        cov = raw_covs[c]

        if mode == 'Tikhonov':
            cov = cov + np.eye(n_features) * regularization_param
        elif mode == 'LDA':
            cov = pooled_cov
        elif mode == 'Friedman':
            S_c = (y_treino == c).sum() * cov
            S_pool = len(y_treino) * pooled_cov
            num = (1 - regularization_param) * S_c + regularization_param * S_pool
            den = (1 - regularization_param) * (y_treino == c).sum() + regularization_param * len(y_treino)
            cov = num / (den + 1e-9)
        elif mode == 'NaiveBayes':
            cov = np.diag(np.diag(cov))
        
        final_covs[c] = cov
        
        try:
            inv_covs[c] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            failed_inversions_count += 1
            inv_covs[c] = np.linalg.pinv(cov) # Usa pseudo-inversa como fallback
      
        sign, log_det = np.linalg.slogdet(cov)
        log_dets[c] = log_det if sign > 0 else -np.inf

    scores = np.zeros((X_teste.shape[0], len(classes)))
    for i, c in enumerate(classes):
        diff = X_teste - means[c]
        mahalanobis = np.sum((diff @ inv_covs[c]) * diff, axis=1)
        scores[:, i] = 0.5 * (mahalanobis + log_dets[c]) - np.log(priors[c] + 1e-9)

    y_pred = classes[np.argmin(scores, axis=1)]
    
    postos = {c: np.linalg.matrix_rank(final_covs[c]) for c in classes}
    model_params = {'m': means, 'S': final_covs, 'posto': postos}
    
    return y_pred, model_params, failed_inversions_count
