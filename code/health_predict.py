# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:52:53 2018

@author: sxx
"""
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint

timesteps_in = 20
timesteps_out = 5
dim_in = 50

def load_data(file):
    xy = np.loadtxt(file, delimiter=',',skiprows=1)
    x = xy[:,8:xy.shape[1]]
    y = xy[:, 0]
    return x,y

def transform_data(x,y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_ = scaler.fit_transform(x)
    y_ = y/100
    return x_, y_

def format_data(x,y):
    dataX = []
    dataY = []
    for i in range(0, len(y) - timesteps_in - timesteps_out):
        _x = x[i:i + timesteps_in]
        _y = y[i + timesteps_in:i + timesteps_in + timesteps_out]  # Next close price
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)   
    return dataX, dataY


def divide_data(x,y):
    train_size = int(len(y) * 0.6)
    valid_size = int(len(y) * 0.8) - train_size 
    test_size = len(y) - valid_size
    trainX, validX, testX = np.array(x[0:train_size]), np.array(x[train_size:
        train_size+valid_size]), np.array(x[train_size+valid_size:len(x)])
    trainY, validY, testY = np.array(y[0:train_size]), np.array(y[train_size:
        train_size+valid_size]),np.array(y[train_size+valid_size:len(y)])
    return trainX,trainY,validX,validY,testX,testY
#记录损失函数的历史数据
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class LstmModel():

    def __init__(self, model=None, epoches=200, batch_size=32, history = LossHistory()):
        self._model = model
        self._epoches = epoches
        self._batch_size = batch_size
        self._history = history
        
    def create_model(self):
        self._model = Sequential()
        self._model.add(LSTM(32, input_shape=(timesteps_in, dim_in), return_sequences=True))
        self._model.add(Dropout(0.3))
        self._model.add(LSTM(8, return_sequences=True))
        self._model.add(Dropout(0.3))
        self._model.add(LSTM(6, return_sequences=False))
        self._model.add(Dropout(0.3))
        self._model.add(Dense(8,activation = 'relu'))
        self._model.add(Dropout(0.3))
        self._model.add(Dense(5))
        #self._model.add(Activation('linear'))
        self._model.add(Activation('sigmoid'))
        self._model.compile(loss='mean_squared_error', optimizer='adam')
        self._model.summary()
        return self._model
    
    def train_model(self, train_x, train_y, valid_x, valid_y, model_save_path):
        self._model.fit(train_x, train_y, epochs=self._epoches, verbose=1, batch_size=self._batch_size, validation_data=(valid_x, valid_y), callbacks=[self._history], shuffle=False)
        self._model.save(model_save_path)
        
        losses = self._history.losses
        return losses

def show_plot(testY,testPredict):
    plt.figure(figsize=(24,48))
    plt.subplot(511)
    plt.plot(testY[0:200,1],c='r',label="train")
    plt.plot(testPredict[0:200,1],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend() 
    
    plt.subplot(512)
    plt.plot(testY[0:200,2],c='r',label="train")
    plt.plot(testPredict[0:200,2],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(513)
    plt.plot(testY[0:200,3],c='r',label="train")
    plt.plot(testPredict[0:200,3],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(514)
    plt.plot(testY[0:200,4],c='r',label="train")
    plt.plot(testPredict[0:200,4],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend()
    
    plt.subplot(515)
    plt.plot(testY[0:200,0],c='r',label="train")
    plt.plot(testPredict[0:200,0],c='g',label="predict")
    plt.xlabel("time aix")
    plt.ylabel("value aix")
    plt.legend() 
    plt.savefig('reslut.png')
    plt.show()
    
if __name__ == '__main__':

    csvfile = './data/health_161.csv'
    x,y = load_data(csvfile)
    x_,y_ = transform_data(x,y)
    datax,datay = format_data(x_,y_)
    trainX,trainY,validX,validY,testX,testY = divide_data(datax,datay)
    '''
    lstmModel = LstmModel()
    lstmModel.create_model()
    save_path = './models/health_0615.h5'
    lstmModel.train_model(trainX,trainY,validX,validY,save_path)
    '''
    print ("predict health score")
    save_path = './models/health_0615.h5'
    loadModel = load_model(save_path)
    testPredict = loadModel.predict(testX)
    show_plot(testY,testPredict)
    print ("end",testPredict.shape)  
