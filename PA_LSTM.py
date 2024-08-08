#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from scipy.fft import fft
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import adfuller
import os
CURRENT_FOLDER  = os.path.dirname(__file__)
SAVEFOLDER      = os.path.join(CURRENT_FOLDER, "result_tuned")
MODELFOLDER     = os.path.join(CURRENT_FOLDER, "pretrain_models")

Tranfer_learning=False


def performance_metric(test, predictions):
    return {
        "rmse": mean_squared_error(test, predictions, squared=False),
        "mse": mean_squared_error(test, predictions, squared=True),
        "mape": mean_absolute_percentage_error(test, predictions),
        "mae": mean_absolute_error(test, predictions)
    }

def read_data(path, rate_train_test=0.75, window=336, period=6):
    print(path)
    df = pd.read_csv(path)
    df = df[['timestamp', 'avgcpu']]
    dataset = df.avgcpu.values  # numpy.ndarray
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * rate_train_test)
    # test sample start after 336*6 of test set
    test_size = len(dataset) - train_size + window*period
    train, test = dataset[0:train_size, :], dataset[-test_size:, :]
    return train, test, scaler, df, train_size

def create_dataset(dataset, window=48, horizon=12, period=2):
    X, Y = [], []
    delta = int(np.ceil(horizon/period))
    # print(delta,horizon, period, len(dataset))
    for i in range(len(dataset)-(window+delta)*period):
        listX = []
        for j in range(window):
            listX.append(dataset[i + j*period, 0])
        X.append(listX)
        # print(i, i + (window+delta)*period, len(dataset))
        Y.append(dataset[i + (window+delta)*period, 0])
    return np.array(X), np.array(Y)

def reshape_dataset(train, test, window=36, horizon=0, period=6):
    look_back = 36
    X_train, Y_train = create_dataset(train, window, horizon, period)
    X_test, Y_test = create_dataset(test, window, horizon, period)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, Y_train, X_test, Y_test

def predict(model, scaler, X_train, Y_train, X_test, Y_test):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = Y_test.reshape(Y_test.shape[0], -1)  # Reshape Y_test to be 2D
    Y_test = scaler.inverse_transform(Y_test)
    return test_predict, Y_test

def saveresult(filename, horizon, window, typepredict, test_predict, Y_test, df):
    len_result = len(test_predict[:, 0])
    df_result = pd.DataFrame(columns=['timestamp', 'avgcpu', 'predict_avgcpu'])
    df_result['timestamp'] = df.timestamp[-len_result:]
    df_result['avgcpu'] = df.avgcpu[-len_result:]
    df_result['predict_avgcpu'] = test_predict[:, 0]
    df_result = df_result.set_index('timestamp')
    pathfile = SAVEFOLDER+filename + "-w" + \
        str(window) + "-h" + str(horizon) + "-" + typepredict+".csv"
    df_result.to_csv(pathfile)
    res = performance_metric(Y_test[:, 0], test_predict[:, 0])
    # typepredict = "LSTM1h_ver0"
    res["filename"] = filename
    res["horizon"] = horizon
    res["window"] = window
    res["typepredict"] = typepredict
    return res
def save_final_result(filename, typepredict, test_predict, Y_test, df):
    len_result = len(test_predict)
    df_result = pd.DataFrame(columns = ['timestamp', 'avgcpu', 'predict_avgcpu'])
    df_result['timestamp'] = df.timestamp[-len_result:]
    # df_result['avgcpu'] = df.avgcpu[-len_result:]
    df_result['avgcpu'] = Y_test[0]
    df_result['predict_avgcpu'] = test_predict
    df_result = df_result.set_index('timestamp')
    pathfile = os.path.join(SAVEFOLDER, filename+ "-" + typepredict+".csv")
    df_result.to_csv(pathfile)
    res = performance_metric(Y_test[0], test_predict[:,0])
    res["filename"]=filename
    path_res = os.path.join(CURRENT_FOLDER, typepredict + "_result_measure.csv")
    red_df = pd.DataFrame(res,index=[0]).to_csv(path_res, mode='a', header=not os.path.exists(path_res) , index=False,columns=["filename", "rmse",  "mse" ,"mape" , "mae"])
    return red_df
def weird_division(n, d):
    return n / d if d else 0

def find_period_fast_fourier(df):
    yf = np.fft.fft(df)
    xf = np.linspace(0.0, 1.0/(2.0), len(df)//2)
    idx = np.argmax(np.abs(yf[1:len(df)//2]))
    freq = xf[idx]
    period = weird_division(1, freq)
    return int(period)

def init_model(X_train):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def retrain_model(modelpath, X_train, Y_train, X_test, Y_test, epochs,newpath):
    if os.path.exists(modelpath):
        model = tf.keras.models.load_model(modelpath)
        print("Checked")
        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False

        # Unfreeze the last two layers
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
    else:
        model = init_model(X_train)
    
    # Define ModelCheckpoint callback
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(newpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_format='tf')
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=256, validation_data=(X_test, Y_test),                       
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    # model.summary()
    return model

def save_result(filename, horizon, window, typepredict, test_predict, Y_test):
    try:
        res = performance_metric(Y_test, test_predict)
        res["filename"]=filename
        res["horizon"]=horizon
        res["window"]=window
        res["typepredict"]=typepredict
        path = typepredict + "_result_measure.csv"
        red_df = pd.DataFrame(res, index=[0]).to_csv(path, mode='a', header=not os.path.exists(path) , index=False,columns=['filename', 'horizon', 'window', "rmse",  "mse" ,"mape" , "mae"])
    # return df_result
    except Exception as e:
        print(e)

if __name__ == "__main__":
    
    list_csv = []
    EPOCHS=100
    n=0
    for week in [1,2,3]:
        rate_train_test = week*0.25
        for file in sorted(os.listdir(INPUTCSV))[:]:
            filename = file.replace(".csv", "")
            windowsize = 48
            horizon = 12
            df = pd.read_csv(INPUTCSV+file)
            period = find_period_fast_fourier(df["avgcpu"])
            print(filename, period)
            if period > 24:
                period = 0
            # if period != 2:
            #     continue
            # if n > 10:
            #     break
            n+=1
            modelpath = os.path.join(MODELFOLDER,"model_LSTM_period_" + str(period))
            print(filename, period, modelpath)
            newpath = MODELFOLDER+"model_LSTM_period_finetuned_" + str(period)
            
            train, test, scaler, df, train_size =read_data(os.path.join(INPUTCSV,file), rate_train_test=rate_train_test)

            X_train, Y_train, X_test, Y_test = reshape_dataset(
                train, test, windowsize, horizon, period)
            print(X_train.shape)
            print(X_test.shape)
            LSTM_model = retrain_model(modelpath, X_train, Y_train, X_test, Y_test, newpath)
            predictions = LSTM_model.predict(X_test, batch_size=256, use_multiprocessing=True)
            test_predict = scaler.inverse_transform(predictions)
            Y_test = scaler.inverse_transform([Y_test])
            
            save_final_result(filename, "LSTM_finetuned_mix_week"+str(week), test_predict, Y_test, df)