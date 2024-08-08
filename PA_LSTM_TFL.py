from  PA_LSTM import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
CURRENT_FOLDER  = os.path.dirname(__file__)
SAVEFOLDER      = os.path.join(CURRENT_FOLDER, "result")
MODELFOLDER     = os.path.join(CURRENT_FOLDER, "models")
INPUTCSV        = os.path.join(CURRENT_FOLDER, "data")
EPOCHS=100
EXPERIMENT_NAME = "LSTM_PA_TFL"
if __name__ == "__main__":
    
    list_csv = []
    EPOCHS=100
    for horizon in [2]:
        
        for file in sorted(os.listdir(INPUTCSV))[:]:
            for week in [1,2,3]:
                rate_train_test = week*0.25
                file_handed = list(pd.read_csv(r"D:\workload\Experiment1\LSTM_PA_TFL_week"+str(week)+"_horizon_2_result_measure.csv").filename)
                # print(file_handed)
                
                filename = file.replace(".csv", "")
                windowsize = 48
                if filename in file_handed:
                    print("Handed:", file)
                    continue
                df = pd.read_csv(INPUTCSV+"//"+file)
                period = find_period_fast_fourier(df["avgcpu"])
                print(filename, period)

                modelpath = os.path.join(MODELFOLDER,"model_LSTM_period_" + str(period)+"_horizon_"+str(horizon))
                print(filename, period, modelpath)
                newpath = MODELFOLDER+"model_LSTM_period_finetuned_" + str(period) + "_horizon_"+str(horizon)
                
                train, test, scaler, df, train_size =read_data(os.path.join(INPUTCSV,file), rate_train_test=rate_train_test)

                X_train, Y_train, X_test, Y_test = reshape_dataset(
                    train, test, windowsize, horizon, period)
                print(X_train.shape)
                print(X_test.shape)
                LSTM_model = retrain_model(modelpath, X_train, Y_train, X_test, Y_test,EPOCHS, newpath)
                predictions = LSTM_model.predict(X_test, batch_size=256, use_multiprocessing=True)
                test_predict = scaler.inverse_transform(predictions)
                Y_test = scaler.inverse_transform([Y_test])
                
                save_final_result(filename, EXPERIMENT_NAME+"_week"+str(week)+"_horizon_"+str(horizon), test_predict, Y_test, df)