import numpy as np
import matplotlib.pyplot as plt


def compute_distances_one_loop(Xtr, xte):
    dist = np.zeros((xte.shape[0], Xtr.shape[0]))
    for i in range(xte.shape[0]):
        d = (Xtr - xte[i]) ** 2
        dist[i, :] = np.sqrt(np.sum(d, axis=1))

    return dist


def compute_distances_no_loop(Xtr, xte):
    # (x-y)^2 = x^2 + y^2 -2xy
    # (n,d) - (x,d) -->(n,1) (x,1) (x,n)
    Xtr_sq = np.sum((Xtr * Xtr), axis=1).reshape(1, -1)  # (1,n)
    Xte_sq = np.sum((xte * xte), axis=1).reshape(-1, 1)  # (x,1)
    dists = np.sqrt(Xte_sq + Xtr_sq - 2 * np.dot(xte, Xtr.T))  # (x,n)

    return dists


def predict(Xtr, ytr, xte, yte, k):
    d = compute_distances_no_loop(Xtr, xte)
    predictions = []
    for i in range(xte.shape[0]):
        closetst_xs = np.argsort(d[i])[:k]
        closest_ys = list(ytr[closetst_xs])
        predicted_y = max(set(closest_ys), key=closest_ys.count)
        predictions.append(predicted_y)
    return np.mean(yte == np.array(predictions))
