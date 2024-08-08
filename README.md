<<<<<<< HEAD
# Boosted Regression for Predicting CPU Utilization in the Cloud with Periodicity
=======
# Boosted Regression for Predicting CPU Utilization in the Cloud with Periodicity

## Introduction
Precise CPU prediction, however, is a tough challenge due to the variable and dynamic nature of CPUs. In this paper, we introduce TrAdaBoost.WLP, a novel regression transfer boosting method that employs Long Short-Term Memory (LSTM) networks for CPU consumption prediction. Concretely, a dedicated periodicity-aware LSTM (PA-LSTM) model is specifically developed to take into account the use of periodically repeated patterns in time series data while making predictions. To adjust for variations in CPU demands, multiple PA-LSTMs are trained and concatenated in TrAdaBoost.WLP using a boosting mechanism. TrAdaBoost.WLP and benchmarks have been thoroughly evaluated on two datasets: 160 Microsoft Azure VMs and 8 Google cluster traces.


## Project Structure

* **data**: contains the input datasets.
* **pretrained_models**: contains the model saved after the pretrain phase.
* **mma_result**: contains the predict result for MIN/MAX/AVG/BEST TradaboostWLP.
* **result**: contains the predict result for TradaboostWLP, TradaBoostR2 and PA-LSTM.

## Install dependencies
```bash
conda create -n workload python=3.7 -y   
conda activate workload               
pip install sklearn==0.0  
pip install arch==5.1.0 keras==2.8.0 matplotlib==3.3.4 numpy==1.21.5 pandas==1.2.3 statsmodels==0.12.2 talos== 1.0.2 tensorflow==2.8.0 tensorflow-gpu==2.8.0 tensorflow-probability==0.14.0
```


## Getting Started
* **Adaboost_WLP_MMA_BEST.py:** It contains Maximum, Minimum, and Average rules make use of PA-LSTMs
* **Adaboost_WLP.py:** It contains the TrAdaBoost.WLP algorithm.
* **PA_LSTM_TFL.py:** It contains the PA-LSTM algorithm with transfers learning.
* **PA_LSTM_non_TFL.py:** It contains the PA-LSTM algorithm without transfers learning.
* **TrAdaboostR2.py:** It contains the TrAdaBoost.R2 algorithm.

## Example command 
```bash
python Adaboost_WLP.py
```
>>>>>>> origin/code
