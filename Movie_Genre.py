import scipy.io
import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from statistics import mean
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m", help="baseline or proposed")
args = parser.parse_args()
if args.mode:
    mode = args.mode
else:
    mode="baseline"
#loading the required datafiles and creating dataframes
Dt_Order_Movie = scipy.io.loadmat('E:/Desktop/FALL 2021/CMPUT 652/Dt_Order_Movie.mat')
Dt_Order_Movie = {k:v for k, v in Dt_Order_Movie.items() if k[0] != '_'}
Df_Order_Movie = pd.DataFrame({k: pd.Series(v[0]) for k, v in Dt_Order_Movie.items()})
Video_Genres = scipy.io.loadmat('E:/Desktop/FALL 2021/CMPUT 652/Video_Genres_mat.mat')

#loading the DCT features 
if(mode =="baseline"):
    Dt_GrnCls_GRDFt = scipy.io.loadmat('E:/Desktop/FALL 2021/CMPUT 652/Dt_GrnCls_GRDFt.mat')
    Df_GrnCls_GRDFt = Dt_GrnCls_GRDFt['DCTFt']
    [subjectNum,clipNum,featureNum] = Df_GrnCls_GRDFt.shape[0],Df_GrnCls_GRDFt.shape[1],Df_GrnCls_GRDFt.shape[2]


target = np.zeros(shape=Video_Genres['Video_Genres_mat'].shape[0])
for i in range(Video_Genres['Video_Genres_mat'].shape[0]):
  target[i]=Video_Genres['Video_Genres_mat'][i][1]

accuracy_list=[]
for i in range(0,subjectNum):
  if mode == "baseline":
    Df_features= pd.DataFrame()
    for j in range(0,clipNum): 
        Df_GrnCls_GRDFt_1 = pd.DataFrame(Df_GrnCls_GRDFt[i][j])
        Df_GrnCls_GRDFt_1= (Df_GrnCls_GRDFt_1.sum(axis=1) - Df_GrnCls_GRDFt_1.mean(axis=1))/Df_GrnCls_GRDFt_1.std(axis=1)
        each_row =(Df_GrnCls_GRDFt_1[:].to_numpy())
        Df_features = Df_features.append(pd.Series(each_row), ignore_index=True)
    
#   elif mode == "proposed":
#         Df_features = pd.read_csv(#PATH TO FILE) there should be 30 csv files (1 for each subject). Each row contains features values for each movie clip 
  
  df_train = Df_features[:30] # 30 movie clips for train
  df_test = Df_features[30:] # 6 movie clips for test
  clf = GaussianNB()
  clf.fit(df_train.fillna(0), target[:30])
  y_pred = clf.predict(df_test.fillna(0))
  y_true = target[30:]
  accuracy= metrics.accuracy_score(y_true, y_pred)
  accuracy_list.append(accuracy)
print("Accuracy:",mean(accuracy_list))
