import time

import numpy as np
from mpmath import norm, eps


def HBA(X, objfunc,  lb, ub,  tmax):
    N, dim = X.shape[0], X.shape[1]
    beta = 6
    NPOP = N
    C = 2
    Food_Score =[]
    CNVG =[]

    vec_flag = np.array([1, - 1])

    # Evaluation
    fitness = objfunc(X)
    GYbest, gbest = np.amin(fitness)
    Xprey = X[gbest,:]
    for t in np.arange(1, tmax + 1).reshape(-1):
        alpha = C * np.exp(- t / tmax)
        I = Intensity(N, Xprey, X)
        for i in np.arange(1, N + 1).reshape(-1):
            r = (np.sqrt(NPOP)/beta)
            F = np.floor(2 * np.random.rand())
            for j in range(dim):
                di = (Xprey[j] - X[i][j])
                if r < 0.5:
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()
                    Xnew = Xprey[j]+ F * beta * I[i]* Xprey[j] + F * r3 * alpha * [di] * np.abs(
                        np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                else:
                    r7 = np.random.rand()
                    Xnew= Xprey(j) + F * r7 * alpha * di
            FU = Xnew> ub
            FL = Xnew< lb
            Xnew[i, :] = (np.multiply(Xnew[i,:], (not (FU + FL)))) + np.multiply(ub, FU) + np.multiply(lb, FL)
            tempFitness = fun_calcobjfunc(objfunc, Xnew[i,:])
            if tempFitness < fitness(i):
                fitness[i] = tempFitness
                X[i, :] = Xnew[i,:]
                FU = X > ub
                FL = X < lb
                X = (np.multiply(X, (not (FU + FL)))) + np.multiply(ub, FU) + np.multiply(lb, FL)
                Ybest, index = np.amin(fitness)
                CNVG = np.amin(Ybest)
                if Ybest < GYbest:
                    GYbest = Ybest
                    Xprey = X[index,:]

                    Food_Score = GYbest
    ct = time.time()
    return Xprey, Food_Score, CNVG,ct


def fun_calcobjfunc(func, X):
    N = X.shape[1 - 1]
    for i in np.arange(1, N + 1).reshape(-1):
        Y = func(X[i, :])

        return Y

def Intensity(N, Xprey, X):
    for i in np.arange( N ):
        di= (norm((X[i,:] - Xprey + eps))) ** 2
        S = (norm((X[i,:] - X[i + 1,:] + eps))) ** 2

    di = (norm((X[N,:] - Xprey + eps))) ** 2
    S= (norm((X[N,:] - X[1,:] + eps))) ** 2
    I=[]
    for i in np.arange(1, N + 1).reshape(-1):
        r2 = np.random.rand()
        I.append(r2 * S(i) / (4 * np.pi * di(i)))

    return I

