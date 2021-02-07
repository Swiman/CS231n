import numpy as np
import pickle
import os


def load_batch(filename):
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") 
        # N,H,W,C = x.shape
        Y = np.array(Y)
        return X, Y


def load(dir):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(dir, "data_batch_%d" % (b,))
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_batch(os.path.join(dir, "test_batch"))
    return Xtr, Ytr, Xte, Yte
