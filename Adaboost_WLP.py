# %%
# coding: UTF-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import adfuller
import os
import time
# from sklearn.tree import DecisionTreeRegressor
CURRENT_FOLDER  = os.path.dirname(__file__)
SAVEFOLDER      = os.path.join(CURRENT_FOLDER, "result")
INPUTCSV        = os.path.join(CURRENT_FOLDER, "data")
# INPUTCSV        = os.path.join(CURRENT_FOLDER, "mix")
MODELFOLDER     = os.path.join(CURRENT_FOLDER, "pretrain_models")
PLOTFOLDER      = os.path.join(CURRENT_FOLDER, "plot")
WINDOWSIZE = 48
LIST_PERIODS = [1,2,3,4,6,12]
RATE_TRAIN_VALIDATE = 0.8
EXPERIMENT_NAME = "Adaboost_100epoch_validate_6iter"
EPOUCH = 200
# INTERATIONS = 10
if not os.path.exists(SAVEFOLDER):
    os.makedirs(SAVEFOLDER)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
def performance_metric(test, predictions):
    try:
        return {    
        "rmse"  : mean_squared_error(test, predictions, squared=False),
        "mse"   : mean_squared_error(test, predictions, squared=True),
        "mape"  : mean_absolute_percentage_error(test, predictions),
        "mae"   : mean_absolute_error(test, predictions)
        }
    except Exception as e:
        print(e)
        return None


# %%
def get_period_data(trans_data, period=1, windowsize=36):
    # transdata is data for training, is a tuple with element is a vector input to model
    # index_dataset is index of element pick from long data, ex period =1, windowsize=4 => array([-4, -3, -2, -1])
    index_dataset = [-(1+i*period) for i in range(windowsize)]
    index_dataset.reverse()
    index_dataset = np.array(index_dataset)
    _trans_data = np.array([np.array(x[index_dataset] ) for x in trans_data])
    return np.reshape(_trans_data, (_trans_data.shape[0], 1, _trans_data.shape[1]))


def save_final_result(filename, horizon, window, typepredict, test_predict, Y_test, df):
    len_result = len(test_predict[0])
    df_result = pd.DataFrame(columns = ['timestamp', 'avgcpu', 'predict_avgcpu'])
    df_result['timestamp'] = df.timestamp[-len_result:]
    # df_result['avgcpu'] = df.avgcpu[-len_result:]
    df_result['avgcpu'] = Y_test[0]
    df_result['predict_avgcpu'] = test_predict[0]
    df_result = df_result.set_index('timestamp')
    pathfile = path_res = os.path.join(SAVEFOLDER, filename+ "-w"+ str(window) + "-h"+ str(horizon) + "-" + typepredict+".csv")
    df_result.to_csv(pathfile)
    res = performance_metric(Y_test[0], test_predict[0])
    # typepredict = "LSTM_adaboost_"
    res["filename"]=filename
    # res["horizon"]=horizon
    # res["window"]=window
    # res["typepredict"]=typepredict
    path_res = os.path.join(CURRENT_FOLDER, typepredict + "_result_measure.csv")
    red_df = pd.DataFrame(res,index=[0]).to_csv(path_res, mode='a', header=not os.path.exists(path_res) , index=False,columns=["filename", "rmse",  "mse" ,"mape" , "mae"])
    return df_result
def save_result_iter(filename, horizon, window, typepredict, test_predict, Y_test, interation, bestmodel, time_inter, error_rate=0):
    res = performance_metric(Y_test[0], test_predict[0])
    # typepredict = "LSTM_adaboost_"
    res["filename"]=filename
    res["horizon"]=horizon
    res["window"]=window
    res["typepredict"]=typepredict
    res["interation"]=interation+1
    res["bestmodel"]=bestmodel
    res['running_time'] = time_inter
    res["error_rate"] = error_rate
    path_res = os.path.join(CURRENT_FOLDER, typepredict + "_interations.csv")
    red_df = pd.DataFrame(res,index=[0]).to_csv(path_res, mode='a', header=not os.path.exists(path_res) , index=False,columns=['filename', 'horizon', 'interation',"bestmodel", 'window', "rmse",  "mse" ,"mape" , "mae", "running_time", "error_rate"])
    return red_df

def read_data(path, rate_train_test=0.75, len_input_vector=48*12, max_line_of_data=None):
    df = pd.read_csv(path)
    df=df[['timestamp','avgcpu']]
    if max_line_of_data and max_line_of_data > len(df):
        df = df.head(max_line_of_data)

    dataset = df.avgcpu.values #numpy.ndarray
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    print("Size dataset:", len(dataset))
    train_size = int(len(dataset) * rate_train_test)
    # test_size = len(dataset) - train_size + window*period # test sample start after 336*6 of test set
    print("Size train_size:", train_size)
    print("Size test_size:", len(dataset) - train_size)
    train, test = dataset[0:train_size,:], dataset[train_size-len_input_vector+1:,:]
    return train, test, scaler, df
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

def reshape_dataset(train, test, window=36, horizon=0,period=6):    
    look_back = 36
    X_train, Y_train = create_dataset(train, window,horizon, period)
    X_test, Y_test = create_dataset(test, window,horizon,period)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, Y_train, X_test, Y_test

def calculate_error_rate(response_R, response_H, weight):
    total = np.abs(response_R - response_H).max()
    return np.sum(weight[:] * np.abs(response_R - response_H) / total)

def TrAdaBoost_WLP(trans_S, Multi_trans_A, response_S, Multi_response_A, test, res_test, scaler, N, filename):


     # prepare trans_A
    trans_A = list(Multi_trans_A.values())[0]
    if len(Multi_trans_A) == 1:
        pass
    else:
        for i in range(len(Multi_trans_A)-1):
            p = i + 1
            trans_A = np.concatenate((trans_A, list(Multi_trans_A.values())[p]), axis=0)
    # prepare response_A
    response_A = list(Multi_response_A.values())[0]
    if len(Multi_response_A) == 1:
        pass 
    else:
        for i in range(len(Multi_response_A)-1):
            p = i + 1
            response_A = np.concatenate((response_A, list(Multi_response_A.values())[p]), axis=0)
   
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_response = np.concatenate((response_A, response_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    if N > row_A:
        print('The maximum of iterations should be smaller than ', row_A)
        
    test_data = np.concatenate((trans_data, test), axis=0)

    # Initialize the weights
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0) 

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # Save prediction response and bata_t
    bata_T = np.zeros([1, N])
    result_response = np.ones([row_A + row_S + row_T, N])

    # Save the prediction response of test data 
    predict = np.zeros([row_T])
    print ('params initial finished.')
    print('='*60)

    trans_data = np.asarray(trans_data, order='C')
    trans_response = np.asarray(trans_response, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        iterstart = time.process_time()
        weights = calculate_P(weights)
        # if i >= 3:
        #     result_response[:, i] = base_regressor(trans_data, trans_response, test_data, weights, remove_worst_model=True)
        # else: 
        result_response[:, i], bestmodel = base_regressor(trans_data, trans_response, test_data, weights, res_test, scaler, i, filename)
        error_rate = calculate_error_rate(response_S, result_response[row_A:row_A + row_S, i],weights[row_A:row_A + row_S, 0])
        # Avoiding overfitting
        if error_rate <= 1e-10 or error_rate > 0.5:
            N = i
            break 
        bata_T[0, i] = error_rate / (1 - error_rate)
        print ('Iter {}-th result :'.format(i))
        print ('error rate :', error_rate, '|| bata_T :', error_rate / (1 - error_rate))
        print('-'*60)

        D_t = np.abs(np.array(result_response[:row_A + row_S, i]) - np.array(trans_response)).max()
        # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], -(np.abs(result_response[row_A + j, i] - response_S[j])/D_t))
        # Changing the data weights of diff-distribution training data
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_response[j, i] - response_A[j])/D_t)

        # #save result at each interation
        # _predict = np.zeros([row_T])
        # for _i in range(row_T):
        #             # predict[i] = 0
        #     M_predict = 0
        #     T_predict = 0
        #     for iter in range(i):
        #         M_predict += np.log(1/bata_T[0, iter]) * result_response[row_A + row_S + _i, iter]
        #         T_predict += np.log(1/bata_T[0, iter])
        #     _predict[_i] = M_predict/T_predict
        #     # _predict[_i] = np.sum(
        #     #     result_response[row_A + row_S + _i, 0:(i+1)]) / (i+1)
        # prediction = scaler.inverse_transform([_predict])
        # Y_test = scaler.inverse_transform([res_test])
        # time_inter = time.process_time() - iterstart
        # res = save_result_iter(filename, 12, WINDOWSIZE, EXPERIMENT_NAME , prediction, Y_test, i, bestmodel, time_inter, error_rate)
    # Save the prediction response of test data 
    predictMIN = np.zeros([row_T])
    predictMAX = np.zeros([row_T])
    predictAVG = np.zeros([row_T])
    predictWLP = np.zeros([row_T])
    predictBEST = np.zeros([row_T])
    print ('params initial finished.')
    print('='*60)
    # WLP caculate
    for i in range(row_T):
        # predict[i] = 0
        M_predict = 0
        T_predict = 0
        for iter in range(N):
            M_predict += np.log(1/bata_T[0, iter]) * result_response[row_A + row_S + i, iter]
            T_predict += np.log(1/bata_T[0, iter])
        predictWLP[i] = M_predict/T_predict
        # predict[i] = np.sum(
        #     # result_response[row_A + row_S + i, int(np.floor(N / 2)):N]) / (N-int(np.floor(N / 2)))
            
        #     result_response[row_A + row_S + i, 0:N]) / (N))
    # Min Max Avg caculate
    for i in range(row_T):
        # predict[i] = 0
        M_predict = []
        T_predict = 0
        for iter in range(N):
            M_predict.append(result_response[row_A + row_S + i, iter])
        predictMIN[i] = np.min(M_predict)
        predictMAX[i] = np.max(M_predict)
        predictAVG[i] = np.average(M_predict)

    # Best case caculate
    best_iter = 0
    for iter in range(N):
        if np.log(1/bata_T[0, iter]) > np.log(1/bata_T[0, best_iter]):
            best_iter = iter
    for i in range(row_T):
        predictBEST[i] = result_response[row_A + row_S + i, best_iter]
            

    # print(predict)
    return predictWLP,predictMIN, predictMAX, predictAVG, predictBEST

def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')

def base_regressor(trans_data, trans_response, test_data, weights, res_test, scaler, iteration, filename, remove_worst_model=False):

    weights = calculate_P(weights)
    cdf = np.cumsum(weights)
    cdf_ = cdf / cdf[-1]
    uniform_samples = np.random.random_sample(len(trans_data))
    bootstrap_idx = cdf_.searchsorted(uniform_samples, side='right')
    bootstrap_idx = np.array(bootstrap_idx, copy=False)
    _trans_data, _trans_response = trans_data[bootstrap_idx], trans_response[bootstrap_idx]
    result = {}
    # split train and validate 
    trainsize = int(len(_trans_data) * RATE_TRAIN_VALIDATE)
    train_X = _trans_data[:trainsize]
    train_Y =  _trans_data[trainsize:]
    validate_X = _trans_response[:trainsize]
    validate_Y = _trans_response[trainsize:]

    # test and transfer with models

    _period = LIST_PERIODS[iteration]
    modelpath = os.path.join(MODELFOLDER,"model_LSTM_period_"+str(_period))
    if os.path.exists(modelpath):
        model = tf.keras.models.load_model(modelpath)
        print("Checked")
        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False

        # Unfreeze the last two layers
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
    # else:
    #     model = init_model()
    _train_X = get_period_data(train_X, period=_period, windowsize=WINDOWSIZE)
    _train_Y = get_period_data(train_Y, period=_period, windowsize=WINDOWSIZE)
    # Define ModelCheckpoint callback
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(modelpath+"_tuned", monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_format='tf')
    
    history = model.fit(_train_X, validate_X, epochs=EPOUCH, batch_size=256, 
                        validation_data=(_train_Y, validate_Y),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=False)
    res_predict = model.predict(get_period_data(test_data, period=_period, windowsize=WINDOWSIZE), use_multiprocessing=True)
    return np.reshape(res_predict, (res_predict.shape[0],)),"model_LSTM_period_"+str(_period)
def read_and_convert_data(train, test, horizon):

    # RATE_TRAIN_VALIDATE = 0.8
    n = int(len(train)*RATE_TRAIN_VALIDATE)
    print("len train test", len(train), len(test))
    train_A_data = train[:n]
    train_S_data = train[-n:]
    trainA, resA =  create_dataset(train_A_data, window=WINDOWSIZE*12, horizon=horizon, period=1)
    trainS, resS =  create_dataset(train_S_data, window=WINDOWSIZE*12, horizon=horizon, period=1)
    train_test, res_test =  create_dataset(test, window=WINDOWSIZE*12, horizon=horizon, period=1)
    Multi_trans_A = {}
    Multi_response_A = {}
    trans_S = trainS
    response_S = resS
    Multi_trans_A["trans_A_1"] = trainA
    Multi_response_A["response_A_1"] = resA
    print("len test", res_test.shape)
    return Multi_trans_A, Multi_response_A, trans_S, response_S, train_test, res_test


    # break


def run_each_week(list_solved_file, horizon):
    # %%
# load dataset 
    # EPOUCH = 2
    # df = pd.read_csv(r"D:\workload\tradaboost_workload\AdaboostV4_epoch100_val_iter10_week1_result_measure.csv")
    # listfilehandled = list(df.filename)
    for file in sorted(os.listdir(INPUTCSV)):
        # for horizon in [1,12,288]:
        # if "period_03" in file:
        # LIST_PERIODS = [1,2,3,4,6,12]
        if file in list_solved_file:
            continue
        filename = file.replace(".csv", "")
        # if filename in listfilehandled:
        #     continue
            # list_csv.append(filename)
    # rate = 0.25, rate train:test = 1:3, max_line_of_data is 4 weekdata
        filepath = os.path.normpath(os.path.join(INPUTCSV,file))

        train, test ,scaler, df  = read_data(filepath, rate_train_test=0.8, len_input_vector=48*12, max_line_of_data=4*7*288)
        # print("1 len train test", len(train),len(test))
        Multi_trans_A, Multi_response_A, trans_S, response_S, train_test, res_test = read_and_convert_data(train, test, horizon)
        predictWLP,predictMIN, predictMAX, predictAVG, predictBEST = TrAdaBoost_WLP(trans_S, Multi_trans_A, response_S, Multi_response_A, train_test, res_test, scaler, len(LIST_PERIODS) , filename)
        prediction = scaler.inverse_transform([predictWLP])
        Y_test = scaler.inverse_transform([res_test])
        res = save_final_result(filename, HORIZON, WINDOWSIZE, "WLP_"+EXPERIMENT_NAME , prediction, Y_test, df)

if __name__ == "__main__":
    
    # print(__file__)
    EPOUCH = 50
    INTERATIONS = 10
    # TEST = True
    HORIZON = 1
    # if TEST:
    #     EPOUCH = 1
    #     INTERATIONS = 2
    for horizon in [2,6,12]:
        HORIZON = horizon     
        
        EXPERIMENT_NAME = "GC_Adaboost" + "_horizon" + str(horizon) 
        if os.path.exists(CURRENT_FOLDER+"\\WLP_"+EXPERIMENT_NAME+ "_result_measure.csv"):
            df = pd.read_csv(CURRENT_FOLDER+"\\WLP_"+EXPERIMENT_NAME+ "_result_measure.csv")
            list_solved_file = []
            for i in list(df.filename):
                list_solved_file.append(i+".csv")
        else:
            list_solved_file = []
        
        run_each_week(list_solved_file, horizon)
        
    # print("Running in:", CURRENT_FOLDER)
    # print("Save in:", SAVEFOLDER)
    # print(os.listdir(INPUTCSV))
