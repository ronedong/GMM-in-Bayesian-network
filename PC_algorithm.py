import pandas as pd
import numpy as np
import pgmpy
from pgmpy.base import DAG
from pgmpy.estimators import PC
from pgmpy.estimators.CITests import pearsonr
import joblib
from sklearn.impute import SimpleImputer
from cdt.data import load_dataset
# data=pd.read_csv('D:/python_work/data/Solar Irradiance Prediction/train.csv')
# data=data.drop(columns=['DATE (YYYY/MM/DD)','MST'])

# data=pd.read_csv('D:/python_work/data/breast cancer/breast_cancer_bulk.csv')
# data=data.drop(columns=['Unnamed: 0'])

data=pd.read_csv('D:/python_work/data/mental health/mhealth_raw_data.csv',nrows=100000)
data=data.drop(columns=['Activity','subject'])
print(data)

# data=pd.read_csv('D:/python_work/data/diamonds.csv',nrows=43152)
# data=data.drop(columns=['color','cut','clarity'])
# print(data)

# data.columns=data_water.columns
# def norm(data):
#     mean=np.mean(data,axis=0)
#     std=np.std(data,axis=0)
#     return (data-mean)/std

# def data_clean(data,n):
#     delete_data=[]
#     for T in data.columns:
#         a=data[abs(data[T])>n]
#         a=list(a.index)
#         delete_data+=a
#     data=data.drop(delete_data)
#     data=data.reset_index(drop=True)
#     return data

# data=norm(data)
# data=data_clean(data,3)


data, graph = load_dataset('sachs')
data=data[:5973]
p=PC(data)
model=p.estimate(ci_test='pearsonr')
print(model.edges())
joblib.dump(model,'D:/python_work/double iteration/PC_DAG_sachs.model')
G=joblib.load('D:/python_work/double iteration/PC_DAG_sachs.model')
print(len(G.edges))