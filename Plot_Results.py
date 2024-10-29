import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable




def plot_results():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 8]
    Algorithm = ['TERMS', 'GWO', 'HBA', 'FHO', 'COA', 'PROPOSED']
    Classifier = ['TERMS', 'DNN', 'RNN-LSTM', 'CNN', 'Without Opt', 'PROPOSED']



    for i in range(Eval.shape[0]):
        value = Eval[i, 4, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------', 'Dataset-',i+1,' - Algorithm Comparison - ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('--------------------------------------------------', 'Dataset-',i+1,'-Classifier Comparison - ',
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_all.npy', allow_pickle=True)
        epoch = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[1], Eval.shape[2]))
            for k in range(Eval.shape[1]):
                for l in range(Eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = Eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(epoch, Graph[:, 0], color='r', linewidth=3, marker='*', markerfacecolor='blue', markersize=16,
                     label="GWO-MDLNet")
            plt.plot(epoch, Graph[:, 1], color='g', linewidth=3, marker='*', markerfacecolor='red', markersize=16,
                     label="HBA-MDLNet")
            plt.plot(epoch, Graph[:, 2], color='b', linewidth=3, marker='*', markerfacecolor='green', markersize=16,
                     label="FHO-MDLNet")
            plt.plot(epoch, Graph[:, 3], color='c', linewidth=3, marker='*', markerfacecolor='cyan', markersize=16,
                     label="COA-MDLNet")
            plt.plot(epoch, Graph[:, 4], color='m', linewidth=3, marker='*', markerfacecolor='black', markersize=16,
                     label="MI-FHCO-MDLNet")
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.xticks(epoch, ('Sigmoid', 'Softamx', 'Relu', 'Tanh', 'LeakyRelu'))
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_1.png" % (i+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()



            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="DNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="RNN-LSTM")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="CNN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="MDLNet")
            ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="MI-FHCO-MDLNet")
            plt.xticks(X + 0.25, ('Sigmoid', 'Softamx', 'Relu', 'Tanh', 'LeakyRelu'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_bar_1.png" % (i+1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()





plot_results()



