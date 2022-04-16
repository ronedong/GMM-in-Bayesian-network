from cdt.causality.graph import GS
import torch
import joblib
import pandas as pd
from cdt.data import load_dataset

# data=pd.read_csv('D:/python_work/data/housing.csv',nrows=16512)
# data=data.drop(columns='Unnamed: 0')

# data=pd.read_csv('D:/python_work/data/diamonds.csv',nrows=1000)
# data=data.drop(columns=['color','cut','clarity'])
# data=pd.DataFrame(data,dtype=float)

# data=pd.read_csv('D:/python_work/data/mental health/mhealth_raw_data.csv',nrows=100000)
# data=data.drop(columns=['Activity','subject'])

data, graph = load_dataset('sachs')
data=data[:5973]

algo=GS()
G=algo.predict(data)
joblib.dump(G,'D:/python_work/double iteration/GS_DAG_sachs.model')
G=joblib.load('D:/python_work/double iteration/GS_DAG_sachs.model')
print(G.edges)