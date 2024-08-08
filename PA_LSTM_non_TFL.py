from  PA_LSTM import *
CURRENT_FOLDER  = os.path.dirname(__file__)
SAVEFOLDER      = os.path.join(CURRENT_FOLDER, "result")
MODELFOLDER     = os.path.join(CURRENT_FOLDER, "models")
INPUTCSV        = os.path.join(CURRENT_FOLDER, "data")
EPOCHS=100
EXPERIMENT_NAME = "LSTM_PA_nonTFL"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == "__main__":
    
    list_csv = []
    EPOCHS=100
    
    for week in [1,2,3]:
        for horizon in [2,6,12]:
            rate_train_test = week*0.25
            for file in sorted(os.listdir(INPUTCSV))[:40]:
                filename = file.replace(".csv", "")
                windowsize = 48
                
                df = pd.read_csv(INPUTCSV+"//"+file)
                period = find_period_fast_fourier(df["avgcpu"])
                print(filename, period)


                modelpath = os.path.join(MODELFOLDER,filename+"_non_model_LSTM_PA_" + str(period)+"_horizon_"+str(horizon))
                print(filename, period, modelpath)
                newpath = os.path.join(MODELFOLDER,filename+"_model_LSTM_PA_" + str(period)+"_horizon_"+str(horizon))
                
                train, test, scaler, df, train_size =read_data(os.path.join(INPUTCSV,file), rate_train_test=rate_train_test)

                X_train, Y_train, X_test, Y_test = reshape_dataset(
                    train, test, windowsize, horizon, period)
                print(X_train.shape)
                print(X_test.shape)
                LSTM_model = retrain_model(modelpath, X_train, Y_train, X_test, Y_test,EPOCHS, newpath)
                predictions = LSTM_model.predict(X_test, batch_size=256, use_multiprocessing=True)
                test_predict = scaler.inverse_transform(predictions)
                Y_test = scaler.inverse_transform([Y_test])
                
                save_final_result(filename, EXPERIMENT_NAME+"_horizon_"+str(horizon), test_predict, Y_test, df)