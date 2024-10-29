import os
import pandas as pd
import numpy as np
from Model_Proposed import Model_Proposed
from Normalize import normalize
from Global_vars import Global_vars
from numpy import matlib
from random import uniform
from GWO import GWO
from HBA import HBA
from FHO import FHO
from COA import COA
from PROPOSED import PROPOSED
from Objective_Fun import Objfun
from Model_DNN import Model_DNN
from Model_LSTM_RNN import Model_LSTM_RNN
from Model_CNN import Model_CNN
# from Plot_Results import plot_results



## READ DATASETS AND PRE-PROCESSING ##
an = 0
if an == 1:
    Datas=[]
    Target=[]
    Path = './Datasets'
    sub_dir = os.listdir(Path)
    for i in range(len(sub_dir)): # For all Datasets
        if 'cardio_train' in sub_dir[i]:
            df = pd.read_csv(Path + '/' + sub_dir[i], delimiter=';')
            # Remove column name 'A'
            df.drop(['id'], axis=1)
            Tar = df['cardio'].to_numpy()
            df.drop(['cardio'], axis=1)
        elif 'heart_disease_uci' in sub_dir[i]:
            df = pd.read_csv(Path + '/' + sub_dir[i])
            df.drop(['id'], axis=1)
            data_top = df.columns
            for j in range(len(data_top)):
                if df[data_top[j]].dtype == 'object':
                    df[data_top[j]] = df['dataset'].astype('category').cat.codes
            Tar = df['num'].to_numpy()
            df.drop(['num'], axis=1)
        elif 'heart_failure' in sub_dir[i]:
            df = pd.read_csv(Path + '/' + sub_dir[i])
            Tar = df['DEATH_EVENT'].to_numpy()
            df.drop(['DEATH_EVENT'], axis=1)
        else:
            df = pd.read_csv(Path + '/' + sub_dir[i])
            Tar = df['target'].to_numpy()
            df.drop(['target'], axis=1)

        df.fillna(value=0, inplace=True) # Fill NAN values
        Data = df.to_numpy()
        uni = np.unique(Tar)
        Targ = np.zeros((len(Tar), len(uni))).astype('int')
        for j in range(len(uni)):
            ind = np.where(Tar == uni[j])
            Targ[ind[0], j] = 1
        Datas.append(Data)
        Target.append(Targ)
    np.save('Datas.npy', Datas)
    np.save('Target.npy', Target)


## DATA TRANSFORMATION ##
an = 0
if an == 1:
    Trans_Data =[]
    Datas = np.load('Datas.npy', allow_pickle=True)
    for i in range(len(Datas)): # For all Datasets
        norm_data = normalize(Datas[i])
        Trans_Data.append(norm_data)
    np.save('Trans_Data.npy', Trans_Data)



## OPTIMIZATION - FOR WEIGHTED FEATURE SELECTION AND FOR Multicascade-DeepLearning ##
an = 0
if an == 1:
    Best_Sol = []
    Data = np.load('Trans_Data.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    for i in range(len(Data)): # For all Datasets
        Global_vars.Features = Data[i]
        Global_vars.Target = Target[i]

        Npop = 10
        Chlen = 5+5  # Number of  values gets Optimized in weighted Feature Selection and for Disease prediction
        xmin = matlib.repmat(np.concatenate([np.zeros((1, 5)), 0.01, 50, 5, 5, 0.01], axis=None), Npop,
                             1)  # 5 for Feature Selection and 2 for epoches and hidden neuron count in RNN and RBF
        xmax = matlib.repmat(np.concatenate([Data[i].shape[1] * np.ones((1, 5)), 0.99, 100, 255, 255, 0.99], axis=None), Npop, 1)
        initsol = np.zeros((xmax.shape))
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objfun
        Max_iter = 25

        print("GWO...")
        [bestfit1, fitness1, bestsol1, time1] = GWO(initsol, fname, xmin, xmax, Max_iter)

        print("HBA...")
        [bestfit2, fitness2, bestsol2, time2] = HBA(initsol, fname, xmin, xmax, Max_iter)

        print("FHO...")
        [bestfit3, fitness3, bestsol3, time3] = FHO(initsol, fname, xmin, xmax, Max_iter)

        print("COA...")
        [bestfit4, fitness4, bestsol4, time4] = COA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed..")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        sols = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Best_Sol.append(sols)
    np.save('Best_Sol.npy', Best_Sol)



## PREDICTION ##
an = 0
if an == 1:
    Eval_all=[]
    Feat = np.load('Trans_Data.npy', allow_pickle=True)
    soln = np.load('Best_Sol.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    vl = [35, 55, 65, 75, 85]
    for n in range(len(Feat)): # For all Datasets
        EV = []
        for m in range(len(vl)):
            per = round(Feat[n].shape[0] * (vl[m] / 100))  # % of learning
            EVAL = np.zeros((10, 14))
            for i in range(5):  # for all algorithms
                sol = soln[n][i]
                Selected_Features = Feat[n][:, sol[:5].astype(int)]
                Weighted_Features = Selected_Features * sol[5]
                train_data = Weighted_Features[:per, :]
                train_target = Target[n][:per, :]
                test_data = Weighted_Features[n][per:, :]
                test_target = Target[per:, :]
                EVAL[i, :] = Model_Proposed(train_data, train_target, test_data, test_target,
                                                  sol[5:])  # Model(With Optimization)
            train_data = Feat[n][:per, :]
            test_data = Feat[n][per:, :]
            EVAL[5, :] = Model_DNN(train_data, train_target, test_data, test_target)  # Model DNN
            EVAL[6, :] = Model_LSTM_RNN(train_data, train_target, test_data, test_target)  # Model RNN-LSTM
            EVAL[7, :] = Model_CNN(train_data, train_target, test_data, test_target)  # Model CNN
            EVAL[8, :] = Model_Proposed(train_data, train_target, test_data, test_target,
                                       np.asarray([23, 25]))  # Model(Without Optimization)
            EVAL[9, :] = EVAL[4, :]
            EV.append(EVAL)
        Eval_all.append(EV)
    np.save('Eval_all.npy', Eval_all)




