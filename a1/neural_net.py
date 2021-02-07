import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        np.random.seed(0)
        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def compute_loss(self, X, y=None, landa=0.0, only_loss=0):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        def f(x): return np.maximum(0, x)
        h1 = f(np.dot(X, W1) + b1)  # (N,H)
        scores = np.dot(h1, W2) + b2
        scores -= np.max(scores, axis=1).reshape(-1, 1)
        e_scores = np.exp(scores)
        softmax_scores = e_scores / np.sum(e_scores, axis=1).reshape(
            -1, 1)  # (N,C)
        losses = np.log(np.sum(e_scores, axis=1)) - scores[np.arange(N), y]
        total_loss = np.mean(losses) + 0.5 * landa * (np.sum(W1**2) +
                                                      np.sum(W2**2))

        if only_loss:
            return total_loss

        grads = {}
        softmax_scores[np.arange(N), y] -= 1
        grads["W2"] = np.dot(h1.T, softmax_scores) / N + landa * W2
        a = np.zeros_like(h1)
        a[h1 > 0] = 1
        dh1 = a * np.dot(softmax_scores, W2.T)
        grads["W1"] = np.dot(X.T, dh1) / N + landa * W1
        grads["b2"] = np.sum(softmax_scores, axis=0) / N
        grads["b1"] = np.sum(dh1, axis=0) / N

        return total_loss, grads

    def compute_loss_analytic(self, X, y, landa, h=0.00001):
        a_grads = {}
        for p in self.params:
            dp = np.zeros_like(self.params[p])
            index = np.nditer(self.params[p],
                              flags=["multi_index"],
                              op_flags=["readwrite"])
            while not index.finished:
                i = index.multi_index
                self.params[p][i] += h
                new_loss = self.compute_loss(X, y, landa, 1)
                self.params[p][i] -= 2 * h
                old_loss = self.compute_loss(X, y, landa, 1)
                dp[i] = (new_loss - old_loss) / (2 * h)
                self.params[p][i] += h
                index.iternext()
            a_grads[p] = dp
        return a_grads

    def train(
            self,
            Xtr,
            ytr,
            Xvl,
            yvl,
            lr=1e-3,
            lr_decay=0.95,
            landa=1e-5,
            num_iters=100,
            batch_size=200,
    ):
        loss_hist = []
        tr_acc_hist = []
        vl_acc_hist = []
        num_tr = Xtr.shape[0]
        step = num_tr // batch_size
        # step = step // 2
        print(step)

        for i in range(num_iters):
            index = np.random.choice(num_tr, batch_size, replace=True)
            xb = Xtr[index]
            yb = ytr[index]
            loss, grads = self.compute_loss(xb, yb, landa)
            loss_hist.append(loss)
            self.params["W1"] -= lr * grads["W1"]
            self.params["b1"] -= lr * grads["b1"]
            self.params["W2"] -= lr * grads["W2"]
            self.params["b2"] -= lr * grads["b2"]

            if i % step == 0:
                tr_acc = (self.predict(Xtr) == ytr).mean()
                vl_acc = (self.predict(Xvl) == yvl).mean()
                tr_acc_hist.append(tr_acc)
                vl_acc_hist.append(vl_acc)
                lr *= lr_decay
                print("%d,loss:%f   tr:%f    vl:%f" % (
                    i,
                    loss,
                    tr_acc,
                    vl_acc,
                ))

        return {
            "loss_hist": loss_hist,
            "tr_acc_hist": tr_acc_hist,
            "vl_acc_hist": vl_acc_hist,
        }
        """
        a = (self.predict(Xtr) == ytr).mean()
        b = (self.predict(Xvl) == yvl).mean()
        return loss, a, b
        """

    def predict(self, X):
        def f(x): return np.maximum(0, x)
        h1 = f(np.dot(X, self.params["W1"]) + self.params["b1"])
        scores = np.dot(h1, self.params["W2"]) + self.params["b2"]
        return np.argmax(scores, axis=1)
