import random
import urllib.request
from os import path
import sys
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from Lib.evaluation import f1
from Lib.mmhc import mmhc


data_alarm=pd.read_csv('D:/python_work/Maximal clique/discrete dataset/alarm.csv',dtype='category',
                        nrows=100000)
data_alarm=data_alarm.drop(columns='Unnamed: 0')

# data_barley=pd.read_csv('D:/python_work/Maximal clique/discrete dataset/barley.csv',dtype='category',
#                         nrows=100000)
# data_barley=data_barley.drop(columns='Unnamed: 0')

data=pd.read_csv('D:/python_work/data/songs.csv',nrows=1000)
data=data.drop(columns='year')
412276


def compare(learning_G,true_G):
    TP=0
    FP=0
    FN=0
    n=len(learning_G.edges)
    for (p,q) in learning_G.edges():
        if (p,q) in true_G.edges or (q,p) in true_G.edges:
            TP+=1
        else:
            FP+=1
    for (p,q) in true_G.edges:
        if (p,q) not in learning_G.edges and (q,p) not in learning_G.edges:
            FN+=1
    return ('TP:',TP/n, 'FP:',FP/n, 'FN:',FN/n)

G=mmhc(data,test ='z-test')
print(G)

# est=MmhcEstimator(data_alarm)
# G=est.estimate()
# print(G.edges)
# joblib.dump(G,'D:/python_work/Maximal clique/MMHC_alarm.model')
# G=joblib.load('D:/python_work/Maximal clique/MMHC_alarm.model')
# print(G)
