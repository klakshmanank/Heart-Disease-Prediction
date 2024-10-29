import numpy as np
import pandas as pd
import cv2 as cv
import tflearn
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from Evaluation import evaluation




def Model_CNN(train_data, Y, test_data, test_y):
    IMG_SIZE = 20
    train_data1 = np.zeros((len(train_data), IMG_SIZE, IMG_SIZE))
    for n in range(len(train_data)):
        train_data1[n, :, :] = cv.resize(train_data[n].astype('float'), (IMG_SIZE, IMG_SIZE))
    X = np.reshape(train_data1, (len(train_data), IMG_SIZE, IMG_SIZE, 1))

    test_data1 = np.zeros((len(test_data), IMG_SIZE, IMG_SIZE))
    for n in range(len(test_data)):
        test_data1[n, :, :] = cv.resize(test_data[n].astype('float'), (IMG_SIZE, IMG_SIZE))
    test_x = np.reshape(test_data1, (len(test_data), IMG_SIZE, IMG_SIZE, 1))


    LR = 1e-3
    ops.reset_default_graph()
    convnet = input_data(shape=[None, 20, 20, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=0.01,
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    pred = np.zeros((test_y.shape[0], test_y.shape[1]))
    model.fit({'input': X}, {'target': Y}, n_epoch=1,
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    out = model.predict(test_x)
    out = np.round(abs(out)).astype('int')
    Eval = evaluation(out.astype('int'), test_y)
    return np.asarray(Eval).ravel()



