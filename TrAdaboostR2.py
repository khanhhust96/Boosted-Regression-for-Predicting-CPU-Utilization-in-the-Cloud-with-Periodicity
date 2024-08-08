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
EXPERIMENT_NAME = "TrAdaBoostR2"
EPOUCH = 100
INTERATIONS = 10
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
# =============================================================================
# Public estimators
# =============================================================================


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
def save_result_eachmodel(filename, typepredict, test_predict, Y_test, interation, modelname, epoch):
    res = performance_metric(Y_test[0], test_predict[:,0])
    # typepredict = "LSTM_adaboost_"
    res["filename"]=filename
    res["epoch"]=epoch
    res["typepredict"]=typepredict
    res["interation"]=interation+1
    res["modelname"]=modelname
    path_res =  os.path.join(CURRENT_FOLDER,typepredict + "_each_iter.csv")
    red_df = pd.DataFrame(res,index=[0]).to_csv(path_res, mode='a', header=not os.path.exists(path_res) , index=False,columns=['filename', 'epoch', 'interation',"modelname", "rmse",  "mse" ,"mape" , "mae"])
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
def create_dataset(dataset, window=336, horizon=12, period=6):
    X, Y = [], []
    for i in range(len(dataset)-window*period-horizon):
        # print("index:",i)
        listX = []
        for j in range(window):
            listX.append(dataset[i + j*period, 0])
        # print(listX)
        X.append(listX)
        Y.append(dataset[i + window*period + horizon, 0])
    return np.array(X), np.array(Y)

def reshape_dataset(train, test, window=36, horizon=0,period=6):    
    look_back = 36
    X_train, Y_train = create_dataset(train, window,horizon, period)
    X_test, Y_test = create_dataset(test, window,horizon,period)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, Y_train, X_test, Y_test
# def predict(model, scaler,X_train, Y_train, X_test, Y_test):
# # def predict(model, scaler,X_train, Y_train, X_test, Y_test):
#     # train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)
#     # # invert predictions
#     # train_predict = scaler.inverse_transform(train_predict)
#     # Y_train = scaler.inverse_transform([Y_train])
#     # test_predict = scaler.inverse_transform(test_predict)
#     # Y_test = scaler.inverse_transform([Y_test])
#     # return test_predict, Y_test
#     return test_predict

def calculate_error_rate(response_R, response_H, weight):
    total = np.abs(response_R - response_H).max()
    return np.sum(weight[:] * np.abs(response_R - response_H) / total)

def TrAdaBoost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, res_test, scaler, N, filename):
    """Boosting for regression transfer. 

    Please feel free to open issues in the Github : https://github.com/Bin-Cao/TrAdaboost
    or 
    contact Bin Cao (bcao@shu.edu.cn)
    in case of any problems/comments/suggestions in using the code. 

    Parameters
    ----------
    trans_S : feature matrix of same-distribution training data

    Multi_trans_A : dict, feature matrix of diff-distribution training data
    e.g.,
    Multi_trans_A = {
    'trans_A_1' :  data_1 , 
    'trans_A_2' : data_2 ,
    ......
    }

    response_S : responses of same-distribution training data, real number

    Multi_response_A : dict, responses of diff-distribution training data, real number
    e.g.,
    Multi_response_A = {
    'response_A_1' :  response_1 , 
    'response_A_2' : response_2 ,
    ......
    }

    test : feature matrix of test data

    N: int, the number of estimators in TrAdaBoost_R2

    Examples
    --------
    # same-distribution training data
    tarin_data = pd.read_csv('M_Sdata.csv')
    # two diff-distribution training data
    A1_tarin_data = pd.read_csv('M_Adata1.csv')
    A2_tarin_data = pd.read_csv('M_Adata2.csv')
    # test data
    test_data = pd.read_csv('M_Tdata.csv')

    Multi_trans_A = {
    'trans_A_1' : A1_tarin_data.iloc[:,:-1],
    'trans_A_2' : A2_tarin_data.iloc[:,:-1]
    }
    Multi_response_A = {
    'response_A_1' :  A1_tarin_data.iloc[:,-1] , 
    'response_A_2' :  A2_tarin_data.iloc[:,-1] ,
    }

    trans_S = tarin_data.iloc[:,:-1]
    response_S = tarin_data.iloc[:, -1]

    test = test_data.iloc[:,:-1]
    N = 20

    TrAdaBoost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, N)

    References
    ----------
    .. [1] section 4.1
    Pardoe, D., & Stone, P. (2010, June). 
    Boosting for regression transfer. 
    In Proceedings of the 27th International Conference 
    on International Conference on Machine Learning (pp. 863-870).
    """

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
    Y_test = scaler.inverse_transform([res_test])

    for i in range(0,N):
        iterstart = time.process_time()
        weights = calculate_P(weights)
        # if i >= 3:
        #     result_response[:, i] = base_regressor(trans_data, trans_response, test_data, weights, remove_worst_model=True)
        # else: 
        result_response[:, i] = base_regressor(trans_data, trans_response, test_data, weights, res_test, scaler, i, filename)
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

         #save result at each interation
        _predict = np.zeros([row_T])
        for _i in range(row_T):
            _predict[_i] = np.sum(
                result_response[row_A + row_S + _i, int(np.floor((i+1)/ 2)):(i+1)]) / (i+1-int(np.floor((i+1) / 2)))
        prediction = scaler.inverse_transform([_predict])
        Y_test = scaler.inverse_transform([res_test])
        time_inter = time.process_time() - iterstart
        res = save_result_iter(filename, 12, WINDOWSIZE, EXPERIMENT_NAME , prediction, Y_test, i, time_inter, error_rate)
    for i in range(row_T):
        predict[i] = np.sum(
            result_response[row_A + row_S + i, int(np.floor(N / 2)):N]) / (N-int(np.floor(N / 2)))


    print("TrAdaBoost_R2 is done")
    print('='*60)
    print('The prediction responses of test data are :')
    print(predict)
    return predict


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

    # _period = LIST_PERIODS[iteration]
    _period = 1
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
    print(res_predict)
    return np.reshape(res_predict, (res_predict.shape[0],))
def read_and_convert_data(train, test, horizon):
    # transform Datan to TrAdaBoost_R2
# in fisrt week, train-validate rate is 0.8 
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


def run(list_solved_file, horizon):
    # %%
# load dataset 
    # print(list_solved_file)
    # EPOUCH = 2
    # df = pd.read_csv(r"D:\workload\tradaboost_workload\AdaboostV4_epoch100_val_iter10_week1_result_measure.csv")
    # listfilehandled = list(df.filename)
    for file in sorted(os.listdir(INPUTCSV)):
        # for horizon in [1,12,288]:
        # if "period_03" in file:
        # LIST_PERIODS = [1,2,3,4,6,12]
        # print(file)
        if file in list_solved_file:
            continue
        print(file)
        filename = file.replace(".csv", "")
        # break
        # if filename in listfilehandled:
        #     continue
            # list_csv.append(filename)
    # rate = 0.25, rate train:test = 1:3, max_line_of_data is 4 weekdata
        filepath = os.path.normpath(os.path.join(INPUTCSV,file))

        train, test ,scaler, df  = read_data(filepath, rate_train_test=0.8, len_input_vector=48*12, max_line_of_data=4*7*288)
    # print("1 len train test", len(train),len(test))
        Multi_trans_A, Multi_response_A, trans_S, response_S, train_test, res_test = read_and_convert_data(train, test,horizon)
        predict = TrAdaBoost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, train_test, res_test, scaler, 10 , filename)
        prediction = scaler.inverse_transform([predict])
        Y_test = scaler.inverse_transform([res_test])
        res = save_final_result(filename, horizon, WINDOWSIZE, EXPERIMENT_NAME , prediction, Y_test, df)

if __name__ == "__main__":
    
    # print(__file__)
    EPOUCH = 100
    INTERATIONS = 10
    # TEST = True
    # if TEST:
    #     EPOUCH = 1
    #     INTERATIONS = 2
    for horizon in [2,6,12]:
        
        EXPERIMENT_NAME = "TrAdaboostR2_GC_" + "_horizon" + str(horizon)
        print(CURRENT_FOLDER+"\\"+EXPERIMENT_NAME+ "_result_measure.csv")
        if os.path.exists(CURRENT_FOLDER+"\\"+EXPERIMENT_NAME+ "_result_measure.csv"):
            df = pd.read_csv(CURRENT_FOLDER+"\\"+EXPERIMENT_NAME+ "_result_measure.csv")
            list_solved_file = []
            for i in list(df.filename):
                list_solved_file.append(i+".csv")
        else:
            list_solved_file = []
        
        run(list_solved_file, horizon)
        # break
    # print("Running in:", CURRENT_FOLDER)
    # print("Save in:", SAVEFOLDER)
    # print(os.listdir(INPUTCSV))
