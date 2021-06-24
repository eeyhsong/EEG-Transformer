"""
Used to calculate the common spatial pattern filter for four-class classification
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


def csp(data_train, label_train):
    idx_0 = np.squeeze(np.where(label_train == 0))
    idx_1 = np.squeeze(np.where(label_train == 1))
    idx_2 = np.squeeze(np.where(label_train == 2))
    idx_3 = np.squeeze(np.where(label_train == 3))

    W = []
    for n_class in range(4):
        if n_class == 0:
            idx_L = idx_0
            idx_R = np.concatenate((idx_1, idx_2, idx_3))
        elif n_class == 1:
            idx_L = idx_1
            idx_R = np.concatenate((idx_0, idx_2, idx_3))
        elif n_class == 2:
            idx_L = idx_2
            idx_R = np.concatenate((idx_0, idx_1, idx_3))
        elif n_class == 3:
            idx_L = idx_3
            idx_R = np.concatenate((idx_0, idx_1, idx_2))

        idx_R = np.sort(idx_R)
        Cov_L = np.zeros([22, 22, len(idx_L)])
        Cov_R = np.zeros([22, 22, len(idx_R)])

        for nL in range(len(idx_L)):
            E = data_train[idx_L[nL], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_L[:, :, nL] = EE / np.trace(EE)
        for nR in range(len(idx_R)):
            E = data_train[idx_R[nR], :, :]
            EE = np.dot(E.transpose(), E)
            Cov_R[:, :, nR] = EE / np.trace(EE)

        Cov_L = np.mean(Cov_L, axis=2)
        Cov_R = np.mean(Cov_R, axis=2)
        CovTotal = Cov_L + Cov_R

        lam, Uc = eig(CovTotal)
        eigorder = np.argsort(lam)
        eigorder = eigorder[::-1]
        lam = lam[eigorder]
        Ut = Uc[:, eigorder]

        Ptmp = np.sqrt(np.diag(np.power(lam, -1)))
        P = np.dot(Ptmp, Ut.transpose())

        SL = np.dot(P, Cov_L)
        SLL = np.dot(SL, P.transpose())
        SR = np.dot(P, Cov_R)
        SRR = np.dot(SR, P.transpose())

        lam_R, BR = eig(SRR)
        erorder = np.argsort(lam_R)
        B = BR[:, erorder]

        w = np.dot(P.transpose(), B)
        W.append(w)

    Wb = np.concatenate((W[0][:, 0:4], W[1][:, 0:4], W[2][:, 0:4], W[3][:, 0:4]), axis=1)
    # The original one is two use the first and last r row, I just use the first 2r.
    # Not significant difference, 2r could be better.

    return Wb


