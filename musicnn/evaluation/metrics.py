import numpy as np
from sklearn.metrics import roc_auc_score


def convert_score_mats_to_pred_list(y_true, y_score, k):
    """Convert score matrice into list of prediction / labels
    
    Args:
        y_true (numpy.ndarray, scipy.sparse.csr_matrix): matrix containing true value
        y_score (numpy.ndarray, scipy.sparse.csr_matrix): matrix containing predicted scores
        k (int): cutoff point
    
    Returns:
        list of list of int: converted list for true labels
        list of list of int: converted list for predicted items
    """
    # get predicted item w.r.t `k`
    # TODO: optimize it using numpy.argpartition
    pred = np.argsort(-y_score, axis=1)[:, :k].tolist()
    true = [
        np.where(y_true[i])[0].tolist()
        for i in range(y_true.shape[0])
    ]
    return true, pred


def ndcg(y_true, y_score, k=None):
    """Normalized discounted cumulative gain at K

    Args:
        y_true (numpy.ndarray): binary matrix of relevant items
        y_score (numpy.ndarray): score matrix of predicted itmes per label
        k (int): threshold to pick relevant items among predicted
                 if it's greater than 1, it works as top-k selection
                 if None, use all the labels to calculate NDCG
    
    Returns:
        float: NDCG at k
    """
    # helper function from
    # https://github.com/eldrin/mf-numba 
    def _ndcg(actual, predicted, k=k):
        if len(predicted) > k:
            predicted = predicted[:k]

        dcg = 0.
        for i, p in enumerate(predicted):
            if np.any(actual == p) and np.all(predicted[:i] != p):
                dcg += 1. / np.log2((i+1) + 1.)

        idcg = np.sum(1. / np.log2(np.arange(1, len(actual)+1) + 1.))
        
        if len(actual) == 0:
            return 0.
        
        return dcg / idcg

    if k is None:
        k = y_true.shape[-1]  # using all values

    true, pred = convert_score_mats_to_pred_list(y_true, y_score, k)
    return np.mean([
        _ndcg(np.array(true[j]), np.array(pred[j]), k)
        for j in range(len(true))
    ])


def apk(y_true, y_score, k=None):
    """Average precision at K

    Args:
        actual (list, set): list of list of integer indice of relevant items
                            (order doesn't matter)
        predicted (list): list of list of integer indice of predicted items
                          (order matters)
        k (int): threshold to pick relevant items among predicted
                 if it's greater than 1, it works as top-k selection
    
    Returns:
        float: average precision truncated at k
    """
    # helper function from
    # https://github.com/eldrin/mf-numba 
    def _apk(actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.
        num_hits = 0.
        
        for i, p in enumerate(predicted):
            if np.any(actual == p) and np.all(predicted[:i] != p):
                num_hits += 1.0
                score += num_hits / (i + 1.)
        
        if len(actual) == 0:
            return 0.
        
        return score / min(len(actual), k)

    true, pred = convert_score_mats_to_pred_list(y_true, y_score, k)
    return np.mean([
        _apk(np.array(true[j]), np.array(pred[j]), k)
        for j in range(len(true))
    ])