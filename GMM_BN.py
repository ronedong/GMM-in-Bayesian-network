import pandas as pd
import numpy as np
import math
from pgmpy.base import DAG
import math
import torch
import joblib
from torch import nn,optim
import itertools
import os
from cdt.data import load_dataset
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#batch=7000 
#house     PC    test_loss=10.0934 10.0729 10.0914  inner_epoch:15  outer_epoch:10 lr=0.005
#house     MMHC  test_loss=10.4105 10.4166 10.3846  inner_epoch:15  outer_epcch:10 lr=0.005
#house     GS    test_loss=9.7039  9.6784  9.6721   inner_epoch:15  outer_epoch:10 lr=0.005
#batch=10000
#mhealth   PC    test_loss=17.8990 17.9058 17.9026 inner_epoch:20  outer_epoch:4  lr=0.005
#mhealth   MMHC  test_loss=17.0573 17.0618 inner_epoch:20  outer_epoch:6  lr=0.005 
#mhealth   GS    test_loss=17.3971 17.3992 inner_epoch:20  outer_epoch:4  lr=0.005
#batch=3000
#sachs     PC    test_loss=14.6697 14.6570 14.6624 inner_epoch:20  outer_epoch:4  lr=0.005 
#sachs     MMHC  test_loss=16.0222 16.0618 16.0431 inner_epoch:20  outer_epoch:5  lr=0.005
#sachs     GS    test_loss=13.6200 13.6093 13.6130 inner_epoch:20  outer_epoch:5  lr=0.005

#diamond   PC    test_loss=6.5766   inner_epoch=30  outer_epoch=5  lr=0.005 weight_decay=1
#diamond   MMHC  test_loss=

# data=pd.read_csv('D:/python_work/data/mental health/mhealth_raw_data.csv',nrows=100000+25000)
# data=data.drop(columns=['Activity','subject'])
# data_train=data[:100000]
# data_test=data[100000:]

# data=pd.read_csv('D:/python_work/data/housing.csv')
# data=data.drop(columns='Unnamed: 0')
# data_train=data[:16512]
# data_test=data[16512:]

data, graph = load_dataset('sachs')
data_train=data[:5973]
data_test=data[5973:]

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
print(data_train)
data_test=norm(data_test)
# data_test=data_clean(data_test,3)
data_test=np.array(data_test)
data_test=torch.Tensor(data_test)


def Linear_Gaussian(T,parents,weights,biase,variance):
    mu=torch.mm(weights,parents.T)+biase
    e=torch.exp(-0.5*(T-mu)**2/variance)
    return 1/((2*np.pi*variance)**(0.5))*e

class Mixture_Linear_Gaussian():
    def __init__(self,T,T_value=0.0,P_value=[],pi=[1.0],W=[],biases=[0.0],Var=[1.0],P=[]):
        self.target_value=T_value
        self.parents_value=P_value
        self.coefficient=pi
        self.weights=W
        self.biases=biases
        self.variances=Var
        self.target=T
        self.parents=P
        self.clusters=0
        
    def pdf(self,i):
        v=self.variances[i]
        w=self.weights[i]
        b=self.biases[i]
        c=self.coefficient[i]
        p_v=self.parents_value[i]
        return c*Linear_Gaussian(self.target_value,p_v,w,b,v)
        
    def pdf_sum(self,pi,W,biases,Var):
        self.coefficient=pi
        self.weights=W
        self.biases=biases
        self.variances=Var
        LGs=[]
        for i in range(self.clusters):
            LGs.append(self.pdf(i))
        return sum(LGs)


def target_cliques(T,PC,variables):
    Cliques=[]
    powerset=[]
    for arr in itertools.permutations(PC[T]):
        Clique={T}
        for X in list(arr):
            if Clique.issubset(set(PC[X])):
                Clique=Clique.union({X})
        if Clique not in Cliques:
            Cliques.append(Clique)
    return Cliques

def to_cliques(G):
    variables=list(G.nodes)
    Cliques=[]
    PC={}
    for T in variables:
        PC[T]=[]
        for X in variables:
            if (T,X) in G.edges or (X,T) in G.edges:
                PC[T].append(X)
    nodes=variables.copy()
    for T in variables:
        Cliques_T=target_cliques(T,PC,variables)
        for clique in Cliques_T:
            if clique not in Cliques:
                Cliques.append(clique)
    return Cliques

                

# def weight_matrix(G,variables):
#     variables=list(variables)
#     n=len(variables)
#     W=torch.zeros((n,n))
#     for i in range(n):
#         A=variables[i]
#         for j in range(n):
#             B=variables[j]
#             if (B,A) in G.edges:
#                 W[i,j]=1
#     return W  

def generate_models(Max_cliques,G,variables):
    models={}
    for i in range(len(variables)):
        T=variables[i]
        model_T=Mixture_Linear_Gaussian(T)
        parents=[]
        weights=[]
        for clique in Max_cliques:
            clique=list(clique)
            if T in clique:
                clique_copy=clique.copy()
                clique_copy.remove(T)
                for X in clique:
                    if (X,T) not in G.edges and X in clique_copy:
                        clique_copy.remove(X)
                if clique_copy!=[] and clique_copy not in parents:
                    parents.append(clique_copy)
                    w=torch.ones((1,len(clique_copy)),requires_grad=True)
                    weights.append(w)
        model_T.weights=weights
        model_T.parents=parents
        if model_T.parents==[]:
            model_T.clusters=1
            model_T.weights=[torch.Tensor([[0.0]])]
        else:
            model_T.clusters=len(parents)
        K=model_T.clusters
        model_T.coefficient=torch.ones(K)/K
        # model_T.coefficient.requires_grad=True
        model_T.biases=torch.zeros(K,requires_grad=True)
        model_T.variances=torch.ones(K,requires_grad=True)
        models[T]=model_T
    return models


def E_step(model,data,k,variables):
    gamma_nk=[]
    M_nk=0
    for m in range(len(data)):
        j=variables.index(model.target)
        model.target_value=data[model.target].loc[m]
        model.parents_value=[]
        if model.parents==[]:
            model.parents_value=torch.Tensor([[0.0]])
        for clique in model.parents:
            parents=[]
            for P in clique:
                l=variables.index(P)
                parents.append(data[P].loc[m])
            parents=torch.Tensor([parents])
            model.parents_value.append(parents)
        gamma_nkm=model.pdf(k)/model.pdf_sum(model.coefficient,model.weights,
                                              model.biases,model.variances)
        M_nk=M_nk+gamma_nkm
    return M_nk

def E_step_tensor(model,data,k,variables):
    gamma_nk=[]
    M_nk=0
    for m in range(len(data)):
        j=variables.index(model.target)
        model.target_value=data[m,j]
        model.parents_value=[]
        if model.parents==[]:
            model.parents_value=torch.Tensor([[0.0]])
        for clique in model.parents:
            parents=[]
            for P in clique:
                l=variables.index(P)
                parents.append(data[m,l])
            parents=torch.Tensor([parents])
            model.parents_value.append(parents)
        gamma_nkm=model.pdf(k)/(model.pdf_sum(model.coefficient,model.weights,
                                              model.biases,model.variances)+1e-6)
        M_nk=M_nk+gamma_nkm
    return M_nk

def loss_function(models,data,variables):
    n=len(variables)
    Likelihood=0
    for i in range(n):
        T=variables[i]
        model_T=models[T]
        # for k in range(len(model_T.parents)):
        #     M_nk=E_step(model_T,data,k,variables)
        #     M=len(data)
        #     model_T.coefficient=np.array(model_T.coefficient)
        #     model_T.coefficient[k]=M_nk/M
        #     model_T.coefficient=torch.Tensor(model_T.coefficient)
        # model_T.coefficient=model_T.coefficient/sum(model_T.coefficient)
        for m in range(len(data)):
            model_T.target_value=data[m,i]
            model_T.parents_value=[]
            if model_T.parents==[]:
                model_T.parents_value=[torch.Tensor([[0.0]])]
            for clique in model_T.parents:
                parents=[]
                for P in clique:
                    j=variables.index(P)
                    parents.append(data[m,j])
                parents=torch.Tensor([parents])
                model_T.parents_value.append(parents)
            # print(model_T.pdf_sum(model_T.coefficient,model_T.weights,
            #                       model_T.biases,model_T.variances))
            Likelihood-=torch.log(model_T.pdf_sum(model_T.coefficient,model_T.weights,
                                                  model_T.biases,model_T.variances)+1e-8)

    Likelihood=Likelihood/len(data)
    return Likelihood

def get_parameters(models,variables):
    parameters=[]
    for T in variables:
        model=models[T]
        parameters.append(model.biases)
        parameters.append(model.variances)
        # if len(model.coefficient)>1:
        #     parameters.append(model.coefficient)
        if model.parents!=[]:
            parameters+=model.weights
    return parameters

def parameters_count(models,variables):
    num=0
    for T in variables:
        model=models[T]
        if model.parents==[]:
            num+=2
        else:
            num+=len(model.biases)
            num+=len(model.variances)
            if len(model.coefficient)>1:
                num+=len(model.coefficient)
            for w in model.weights:
                num+=len(w)
    return num

def training_model(data,models,optimizer,n_batch,n_epochs,variables):
    for epoch in range(n_epochs):
        batch=data.sample(n=n_batch,replace=False)
        batch=np.array(batch)
        batch=torch.Tensor(batch)
        # for T in variables:
        #     model=models[T]
        #     for k in range(len(model.parents)):
        #         M_nk=E_step(model,batch,k,variables)
        #         M=n_batch
        #         model.coefficient[k]=M_nk/M
        # for T in variables:
        #     model=models[T]
        #     model.coefficient=model.coefficient/sum(model.coefficient)
        loss=loss_function(models,batch,variables)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:',epoch)
        print('Loss:',loss)
    #     if loss==None:
    #         for T in variables:
    #             model=models[T]
    #             model.coefficient=model.coefficient/sum(model.coefficient)
    #         return models    
    for T in variables:
        model=models[T]
        model.coefficient=model.coefficient/sum(model.coefficient)
    return models

def double_iteration(data,models,optimizer,n_batch,inner_epochs,outer_epochs,variables):
    n=len(variables)
    for out_epoch in range(outer_epochs):
        # outer_batch=data.sample(n=m_batch,replace=False)
        # outer_batch=outer_batch.reset_index(drop=True)
        for i in range(n):
            T=variables[i]
            model_T=models[T]
            for k in range(len(model_T.parents)):
                M_nk=E_step(model_T,data,k,variables)
                M=len(data)
                model_T.coefficient=np.array(model_T.coefficient)
                model_T.coefficient[k]=M_nk/M
                model_T.coefficient=torch.Tensor(model_T.coefficient)
        print('outer Epoch:',out_epoch)
        # if out_epoch==5:
        #     inner_epochs=35
        for in_epoch in range(inner_epochs):
            batch=data.sample(n=n_batch,replace=False)
            batch=np.array(batch)
            batch=torch.Tensor(batch)
            loss=loss_function(models,batch,variables)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('inner_Epoch:',in_epoch)
            print('Loss:',loss)
            # if in_epoch==19 or in_epoch==9:
            #     print('Test loss:',loss_function(models,data_test,variables))
    # for T in variables:
    #     model=models[T]
    #     model.coefficient=model.coefficient/sum(model.coefficient)
    return models


G=joblib.load('D:/python_work/double iteration/GS_DAG_sachs.model')
variables=list(data_train.columns)
Max_cliques=to_cliques(G)
models=generate_models(Max_cliques,G,variables)
# print(len(variables))
# print(len(G.edges))
# print(parameters_count(models, variables))

optimizer=optim.Adam(get_parameters(models,variables),lr=0.005)

# models=double_iteration(data_train,models,optimizer,3000,20,4,variables)
# joblib.dump(models,'D:/python_work/double iteration/GS_GMM_mhealth.model')
# models=joblib.load('D:/python_work/double iteration/GS_GMM_.model')

print(loss_function(models,data_test,variables))



# for T in variables:
#     model=models[T]
#     print(model.parents)
#     print(model.weights)
#     print(model.biases)
#     print(model.variances)
#     print('pi:',model.coefficient)