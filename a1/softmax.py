import numpy as np


def compute_loss_naive(X, y, W, landa):
    dw = np.zeros_like(W)
    total_loss = 0.0
    for i in range(X.shape[0]):
        scores_i = np.dot(X[i], W)
        scores_i -= np.max(scores_i)
        e_scores_i = np.exp(scores_i)
        loss_i = -scores_i[y[i]] + np.log(np.sum(e_scores_i))
        total_loss += loss_i
        scores_i = e_scores_i / np.sum(e_scores_i)
        for j in range(W.shape[1]):
            if j == y[i]:
                scores_i[j] -= 1
            dw[:, j] += scores_i[j] * X[i]
    dw = dw / X.shape[0] + landa * W
    total_loss /= X.shape[0]
    total_loss += 0.5 * landa * np.sum(W**2)
    return dw, total_loss


def compute_loss(X, y, W, landa):
    scores = np.dot(X, W)  # (n,d)*(d,c)->(n,c)
    scores -= np.max(scores, axis=1).reshape(-1, 1)
    e_scores = np.exp(scores)
    losses = -scores[np.arange(X.shape[0]), y] + np.log(
        np.sum(e_scores, axis=1))
    total_loss = np.mean(losses)
    scores = e_scores / np.sum(e_scores, axis=1).reshape(-1, 1)
    scores[np.arange(X.shape[0]), y] -= 1  # (n,c)
    dw = np.dot(X.T, scores)  # (d,n)*(n,c) -> (d,c)
    dw = dw / X.shape[0] + landa * W
    total_loss += 0.5 * landa * np.sum(W**2)
    return dw, total_loss


def train(X, y, W, landa, b_size, lr, epoch):
    loss_hist = []
    for i in range(epoch):
        indexes = np.random.choice(X.shape[0], b_size, replace="false")
        X_batch = X[indexes]
        y_batch = y[indexes]
        dw, loss = compute_loss(X_batch, y_batch, W, landa)
        loss_hist.append(loss)
        W -= lr * dw
    return loss_hist, W


def predict(X, y, W):
    scores = np.dot(X, W)
    predicted_ys = np.argmax(scores, axis=1)
    return predicted_ys
