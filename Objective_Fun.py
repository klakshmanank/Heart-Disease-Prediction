import numpy as np
import cv2 as cv
from Global_vars import Global_vars
from Model_Proposed import Model_Proposed


def Objfun(soln):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    for k in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = soln[k,:]
        else:
            sol = soln

        Selected_Features = Global_vars.Features[:, sol[:5].astype('int')]
        Weighted_Features = Selected_Features * sol[5]
        Target = Global_vars.Target
        learnper = round(Weighted_Features.shape[0] * 0.75)
        train_data = Weighted_Features[learnper:, :]
        train_target = Target[learnper:, :]
        test_data = Weighted_Features[:learnper, :]
        test_target = Target[:learnper, :]

        Eval = Model_Proposed(train_data, train_target, test_data, test_target, sol[5:].astype('int'))  # Model AutoEncoder-TransLSTM-DBN
        fitn[k] = 1 / (Eval[0][4])

    return fitn
