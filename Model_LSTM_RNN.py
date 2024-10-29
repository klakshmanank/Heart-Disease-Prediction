import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from Evaluation import evaluation




def Model_LSTM_RNN(trainX, trainY, testX, test_y):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    testPredict = np.zeros((test_y.shape[0], test_y.shape[1])).astype('int')
    for i in range(trainY.shape[1]):
        model.fit(trainX, trainY[:, i], epochs = 1, batch_size=1, verbose=2)
        model.fit(trainX, trainY[:, i], epochs=1, batch_size=1, verbose=2)
        testPredict[:, i] = model.predict(testX).ravel()
    predict = np.round(testPredict).astype('int')
    Eval = evaluation(predict, test_y)
    return np.asarray(Eval).ravel()




