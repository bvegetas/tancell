#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 01:53:09 2019

@author: bvegetas
"""
import numpy as np

class DiscreteUniqueCode:
    def __init__(self):
        self.n_dim=0
        
    def fit(self,data):
        self.n_dim=data.shape[1]
        self.glossary=np.zeros(self.n_dim).tolist()
        for i in range(self.n_dim):
            self.glossary[i]=np.unique(data[:,i])
        return self
    def predict(self,data):
        code=np.zeros(data.shape).astype(int)
        for i in range(self.n_dim):
            code[:,i]=len(self.glossary[i])
            for j in range(len(self.glossary[i])):
                code[data[:,i]==self.glossary[i][j],i]=j
            
        return code
    def sample(self,code):
        #data=np.chararray(code.shape,256,unicode=True)
        data=np.zeros(code.shape).astype(np.dtype('<U256'))
        for i in range(self.n_dim):
            for j in range(len(self.glossary[i])):
                data[code[:,i]==j,i]=self.glossary[i][j]
            
        return data
        