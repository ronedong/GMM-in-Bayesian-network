import pandas as pd
import numpy as np
import pgmpy
from pgmpy.factors.continuous import LinearGaussianCPD
import math
from pgmpy.base import DAG
import math
import torch
import joblib
from torch import nn,optim
import os
from cdt.data import load_dataset
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#house       
#PC   test_loss=13.1489 13.1482 13.1482 epoch:70  lr=0.005 
#GS   test_loss=13.0906 13.0910 13.1032 epoch:70  lr=0.005
#sachs       
#PC   test_loss=16.0398 16.0273 16.0296 epoch:60  lr=0.005 weight_decay=1
#GS   test_loss=15.8411 15.8392 15.8415 epoch:50

data=pd.read_csv('D:/python_work/data/housing.csv')
data=data.drop(columns='Unnamed: 0')
data_train=data[:16512]
data_test=data[16512:]

# data=pd.read_csv('D:/python_work/data/mental health/mhealth_raw_data.csv',nrows=100000+25000)
# data=data.drop(columns=['Activity','subject'])
# data_train=data[:100000]
# data_test=data[100000:]

# data=pd.read_csv('D:/python_work/data/diamond.csv')
# data=data.drop(columns=['color','cut','clarity'])
# data_train=data[:43152]
# data_test=data[43152:]

# data, graph = load_dataset('sachs')
# data_train=data[:5973]
# data_test=data[5973:]

def norm_tensor(data):
    mean=torch.mean(data,dim=0)
    std=torch.std(data,dim=0)
    return (data-mean)/std

def norm(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    return (data-mean)/std

def data_clean(data,n):
    delete_data=[]
    for T in data_train.columns:
        a=data[abs(data[T])>n]
        a=list(a.index)
        delete_data+=a
    data=data.drop(delete_data)
    data=data.reset_index(drop=True)
    return data

data_train=norm(data_train)
data_train=data_clean(data_train,3)

data_test=norm(data_test)
# data_test=data_clean(data_test,3)
data_test=np.array(data_test)
data_test=torch.Tensor(data_test)

sigmoid=nn.Sigmoid()
def log_Linear_Gaussian(T,parents,weights,biase,variance):
    mu=sigmoid(torch.mm(weights,parents.T))+biase
    e=-0.5*(T-mu)**2/variance
    return -0.5*torch.log(2*np.pi*variance)+e

class Linear_Gaussian():
    def __init__(self,T,T_value=0.0,P_value=[],W=[0.0],biase=[0.0],Var=[1.0],P=[]):
        self.target_value=T_value
        self.parents_value=P_value
        self.weights=W
        self.biase=biase
        self.variance=Var
        self.target=T
        self.parents=P
    def log_Linear_Gaussian(self,T_value,P_value,W,biase,Var):
        return log_Linear_Gaussian(T_value,P_value,W,biase,Var)

def generate_models(G,variables):
    variables=list(variables)
    n=len(variables)
    models={}
    for T in variables:
        model=Linear_Gaussian(T)
        model.parents=[]
        for X in variables:
            j=variables.index(X)
            if (X,T) in G.edges:
                model.parents.append(X)
        model.weights=torch.ones((1,len(model.parents)),requires_grad=True)
        model.biase=torch.tensor([[0.0]],requires_grad=True)
        model.variance=torch.tensor([[1.0]],requires_grad=True)
        models[T]=model
    return models

def weight_matrix(models,variables):
    n=len(models)
    W=torch.zeros((n,n))
    for T in variables:
        model=models[T]
        i=variables.index(T)
        for P in model.parents:
            j=variables.index(P)
            k=model.parents.index(P)
            W[i,j]=model.weights[0,k]
    return W

def loss_function(models,data,variables):
    Likelihood=0
    for T in variables:
        model=models[T]
        i=variables.index(T)
        for m in range(len(data)):
            model.target_value=data[m,i]
            model.parents_value=[]
            for P in model.parents:
                j=variables.index(P)
                model.parents_value.append(data[m,j])
            model.parents_value=torch.tensor([model.parents_value])
            Likelihood-=model.log_Linear_Gaussian(model.target_value,model.parents_value,
                                                  model.weights,model.biase,
                                                  model.variance)
    Likelihood=Likelihood/len(data)
    return Likelihood

def Bayesian_score(models,data,variables):
    N=0
    for T in variables:
        model=models[T]
        N+=len(model.weights)+3
    Likelihood=loss_function(models,data,variables)
    BS=Likelihood*len(data)+N*torch.log(len(data))
    return BS

def get_parameters(models,variables):
    parameters=[]
    for T in variables:
        model=models[T]
        parameters.append(model.weights)
        parameters.append(model.biase)
        parameters.append(model.variance)
    return parameters
 
def training_model(data,models,optimizer,n_batch,n_epochs,variables):
    for epoch in range(n_epochs):
        batch=data.sample(n=n_batch,replace=False)
        batch=np.array(batch)
        batch=torch.Tensor(batch)
        loss=loss_function(models,batch,variables)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:',epoch)
        print('Loss:',loss)
        if (epoch+1)%10==0:
            print('Test Loss:',loss_function(models,data_test,variables))
    return models

# 15.95 130 1

G=joblib.load('D:/python_work/double iteration/GS_DAG_house.model')
variables=list(data_train.columns)
models=generate_models(G,variables)
print(G.edges)
optimizer=optim.Adam(get_parameters(models,variables),lr=0.005)
models=training_model(data_train,models,optimizer,7000,70,variables)

joblib.dump(models,'D:/python_work/double iteration/GS_LG(sigmoid)_house.model')
models=joblib.load('D:/python_work/double iteration/GS_LG(sigmoid)_house.model')
print(loss_function(models,data_test,variables))
# for T in variables:
#     model=models[T]
#     print(model.target)
#     print(model.parents)
#     print(model.variance)

# parameters_number=0
# for p in get_parameters(models, variables):
#     parameters_number+=len(p)
# print(parameters_number)