import numpy as np


def compute_loss_naive(X, y, w, landa):
    # scores = np.dot(X, w)  # (500,3073) * (3073,10)
    loss = 0.0
    dw = np.zeros_like(w)
    for i in range(X.shape[0]):
        scores = np.dot(X[i], w)
        margins = np.maximum(0, scores - scores[y[i]] + 1)
        margins[y[i]] = 0
        loss_i = 0.0
        for j in range(w.shape[1]):
            if j == y[i]:
                continue
            if margins[j] > 0:
                dw[:, j] += X[i].T
                dw[:, y[i]] -= X[i].T
                loss_i += margins[j]
        loss += loss_i

    total_loss = loss / X.shape[0]
    total_loss += 0.5 * landa * np.sum(w ** 2)

    dw = dw / X.shape[0] + landa * w
    return total_loss, dw


def compute_loss(X, y, w, landa):
    scores = np.dot(X, w)  # (500,3073) * (3073,10)
    correct_scores = scores[np.arange(X.shape[0]), y].reshape(-1, 1)  # (500,1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(X.shape[0]), y] = 0
    losses = np.sum(margins, axis=1)
    total_loss = np.mean(losses)
    total_loss += 0.5 * landa * np.sum(w ** 2)

    dw = np.zeros_like(w)
    coeff_matrix = np.zeros_like(margins)
    coeff_matrix[margins > 0] = 1
    coeff_matrix[np.arange(X.shape[0]), y] = -np.sum(coeff_matrix, axis=1)
    dw = np.dot(X.T, coeff_matrix)  # (500,10) (3073,500)
    dw = dw / X.shape[0] + landa * w
    return total_loss, dw


def train(X, y, batch_size, landa, lr, epochs):
    num_classes = max(y) + 1
    D = X.shape[1]
    np.random.seed(0)
    W = 0.0001 * np.random.randn(D, num_classes)
    losses = []
    tr_accs = []
    for i in range(epochs):
        indexes = np.random.choice(X.shape[0], batch_size, replace="false")
        Xdv = X[indexes]
        ydv = y[indexes]
        loss, grad = compute_loss(Xdv, ydv, W, landa)
        losses.append(loss)
        W -= lr * grad
        tr_accs.append(predict(X, y, W))
    return W, losses, tr_accs


def predict(X, y, w):
    scores = np.dot(X, w)
    predicted_ys = np.argmax(scores, axis=1)
    return np.mean(predicted_ys == y)

