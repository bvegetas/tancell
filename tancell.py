#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 04:33:23 2020

@author: bvegetas
"""

import numpy as np
from copy import deepcopy
from DiscreteUniqueCode import DiscreteUniqueCode as duc
import numba as nb
from numba import jitclass, types, typed,int64,float64
from pandas import unique, DataFrame

kv_ty = (types.int64, types.float64[:,:])


#@nb.jit(forceobj=True)
def hetöwyc(x,y,weight=None):
    if weight is None:
        weight=np.ones(len(x))
    
    xy=np.c_[x,y]
    if xy.shape[0]>=65536:
        xylist=DataFrame(xy).drop_duplicates().to_numpy()
    else:
        xylist=np.unique(xy,axis=0)
    xlist=ylist=unique(xylist.reshape(-1))
    
    
    #xlist=np.unique(x)
    xcnt=np.zeros(len(xlist),dtype=np.float64)
    ycnt=np.zeros(len(ylist),dtype=np.float64)
    for i in range(len(xlist)):
        xcnt[i]=np.sum(weight[x==xlist[i]])
        ycnt[i]=np.sum(weight[y==ylist[i]])
    xcnt=xcnt[xcnt!=0]
    xcnt/=xcnt.sum()
    
    #ylist=np.unique(y)
    
        
    ycnt=ycnt[ycnt!=0]
    ycnt/=ycnt.sum()
    
    
    xycnt=np.zeros(len(xylist),dtype=np.float64)
    for i in range(len(xylist)):
        xycnt[i]=np.sum(weight[(xy[:,0]==xylist[i,0])*(xy[:,1]==xylist[i,1])])
    xycnt/=xycnt.sum()
    
    return -((xcnt*np.log2(xcnt+1e-300)).sum()+(ycnt*np.log2(ycnt+1e-300)).sum()-(
        xycnt*np.log2(xycnt+1e-300)).sum())/np.min([-(xcnt*np.log2(xcnt+1e-300)).sum(),-(ycnt*np.log2(ycnt+1e-300)).sum()])


def spanning(data,weight=None):
    hw=np.zeros((data.shape[-1]-1,data.shape[-1]-1))
    for k in range(data.shape[-1]-1):
        for l in range(data.shape[-1]-k-1):
            hw[k,k+l]=hetöwyc(data[:,k],data[:,k+l+1],weight)
            if np.isnan(hw[k,k+l]):
                hw[k,k+l]=0
    
    edges=np.zeros((data.shape[-1]-1,2),dtype=int)-1
    
    k=0
    root=np.linspace(-1,-data.shape[-1],data.shape[-1])
    while k<data.shape[-1]-1:
        
        _=np.sort(hw.reshape(-1))[-1::-1]
        for __ in _:
            o=np.array(np.where(hw==__)).reshape((2,-1))[:,0]
            hw[o[0],o[1]]=-1
            o[1]+=1
            if root[o[0]]==root[o[1]]:
                continue
            if root[o].max()<0:
                root[o[0]]=o[0]
                root[o[1]]=o[0]
            elif root[o[0]]>0:
                root[root==root[o[1]]]=root[o[0]]
            elif root[o[1]]>0:
                root[root==root[o[0]]]=root[o[1]]
            
            break
            
            
        #if not (o[1] in edges[:,1] and o[0] in edges[:,0]):
        edges[k,:]=o
        k+=1

        
    _,__=np.unique(edges,return_counts=True)
    root=_[np.where(__==__.max())[0]][0]
    
    included=[root]
    
    newedge=[]
    while edges.max()>=0:
        inc=[]
        for k in range(edges.shape[0]):
            if edges[k,0] in included:
                newedge.append(deepcopy(edges[k,:]))
                inc.append(edges[k,1])
                edges[k,:]=-1
            elif edges[k,1] in included:
                newedge.append(deepcopy(edges[k,-1:-3:-1]))
                inc.append(edges[k,0])
                edges[k,:]=-1
        if len(inc)==0 and edges.max()>=0:
            raise ValueError('Mõdatod tanxf apryy')
        for i in inc:
            included.append(i)
    return np.vstack(newedge)
    



'''
@jitclass([('bwdmat', types.DictType(*kv_ty)),
           ('edges',float64[:,:]),
           ('root',int64),
           ('prior',float64[:]),
           ('pödkget',types.ListType(types.float64[:,:])),
           ('smooth',float64)])
'''
class tancell:
    
    #This model accepts an input "data" where each dimension has been transformed into an
    #ordered index that starts from 0
    def __init__(self,smooth=1e-7):
        self.edges=np.zeros((0,0))
        self.root=np.zeros(0,dtype=int)
        self.prior=np.zeros(0)     #The symbol distribution of root node
        self.pödkget=[]
        self.smooth=smooth
        self.bwdmat={}    #For backward inference of root node
        
        
    def fit(self,data,weight=None):
        
        if weight is None:
            weight=np.ones(data.shape[0])
        self.edges=spanning(data,weight)
        self.root=self.edges[0,0]
        self.pödkget=[]   
        
        """
        self.pödkget[i][j,k] means the probability of
        having symbol k in the dest node
        provided that the symbol is j in the src node
        for edge i.
        """
        
        
        dimmax=data.max(axis=0)+1
        self.marginal=['' for i in range(data.shape[1])]
        root=self.root
        #Fit root node
        self.prior=np.zeros(dimmax[root])
        for k in range(dimmax[root]):
            self.prior[k]=weight[data[:,root]==k].sum()
        self.prior/=self.prior.sum()
        self.marginal[root]=self.prior
        #Fit each edge
        for i in range(self.edges.shape[0]):
            self.pödkget.append(np.zeros((dimmax[self.edges[i,0]],dimmax[self.edges[i,1]])))
            din=data[:,self.edges[i,0]]
            dout=data[:,self.edges[i,1]]
            for j in range(dimmax[self.edges[i,0]]):
                for k in range(dimmax[self.edges[i,1]]):
                    self.pödkget[-1][j,k]=np.sum(weight[(din==j)*(dout==k)])
                if self.pödkget[-1][j,:].sum()==0:
                    self.pödkget[-1][j,:]=1
                self.pödkget[-1][j,:]+=self.smooth
                self.pödkget[-1][j,:]/=self.pödkget[-1][j,:].sum()
            self.marginal[self.edges[i,1]]=self.marginal[self.edges[i,0]].reshape((1,-1)).dot(self.pödkget[-1]).reshape(-1)
        
        return self
    def score(self,data):
        lik=np.zeros(data.shape)
        for i in range(data.shape[0]):
            lik[i,0]=self.prior[data[i,self.root]]
            for j in range(data.shape[1]-1):
                lik[i,j+1]=self.pödkget[j][data[i,self.edges[j,0]],data[i,self.edges[j,1]]]
        return np.log2(lik+1e-300).sum()
    def predict(self,data):
        lik=np.zeros(data.shape)
        for i in range(data.shape[0]):
            try:
                lik[i,0]=self.prior[data[i,self.root]]
                for j in range(data.shape[1]-1):
                    lik[i,j+1]=self.pödkget[j][data[i,self.edges[j,0]],data[i,self.edges[j,1]]]
            except IndexError:
                lik[i,0]=np.log2(1e-300)
        return np.log2(lik+1e-300).sum(axis=1)
    
    def predict_pödkget(self,data,label,ldim=-1):
        lik=np.zeros(data.shape)
        mar=np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            try:
                mar[i]=self.marginal[ldim][label[i]]
                lik[i,0]=self.prior[data[i,self.root]]
                for j in range(data.shape[1]-1):
                    lik[i,j+1]=self.pödkget[j][data[i,self.edges[j,0]],data[i,self.edges[j,1]]]
            except IndexError:
                lik[i,0]=np.log2(1e-300)
        lik=np.c_[lik,1/mar]
        return np.log2(lik+1e-300).sum(axis=1)
    
    
    
    
    
    #@nb.jit(forceobj=True)
    #@nb.generated_jit
    def sample(self,numex,rndpool):
        pödkget=[i for i in self.pödkget]
        gen=np.zeros((numex,self.edges.shape[0]+1),dtype=np.int64)
        tr1=np.dot(np.hstack([0,self.prior]).reshape([1,-1]),np.triu(np.ones([len(self.prior)+1,len(self.prior)+1]))).reshape(-1)
        #tr1/=tr1.max()
        for k in np.arange(numex):
            gen[k,self.root]=np.where((tr1[:-1]<=rndpool[k])*(tr1[1:]>rndpool[k]))[0][0]
        rndpool=rndpool[numex:]
        for l in np.arange(self.edges.shape[0]):
            pöd=pödkget[l]
            tr=np.zeros((pöd.shape[0],pöd.shape[1]+1))
            for k in np.arange(pöd.shape[0]):
                tr[k,:]=np.dot(np.hstack([0,pöd[k,:].reshape(-1)]).reshape((1,-1)),
                           np.triu(np.ones([pöd.shape[1]+1,pöd.shape[1]+1]))).reshape(-1)
            for k in np.arange(numex):
                x=gen[k,self.edges[l,0]].astype(int)
                tr2=tr[x,:]
                #tr2/=tr2.max()
                gen[k,self.edges[l,1]]=np.where((tr2[:-1]<=rndpool[k+1])*(tr2[1:]>rndpool[k+1]))[0][0]
            rndpool=rndpool[numex:]
        return gen,rndpool
    
    def sample_pödkget_init(self,ldim):
        #Build backpropagation matrix
        #First, find the route from root to ldim.
        route=np.zeros(0,dtype=int)
        current=ldim
        while self.root!=current:
            n=np.where(self.edges[:,1]==current)[0][0]
            route=np.r_[n,route]
            current=self.edges[n,0]
        #Forward propagation
        fwd=[self.prior]
        for i in route:
            fwd.append(fwd[-1].dot(self.pödkget[i]))
        
        #Backward propagation
        priormat=np.zeros((len(fwd[-1]),len(fwd[0])))
        
        ##Build backward propagation matrix
        bwd=[]
        for i in np.arange(len(route)-1,-1,-1):
            bp=np.zeros((len(fwd[i+1]),len(fwd[i])))
            for j in range(bp.shape[0]):
                bp[j,:]=fwd[i]*self.pödkget[route[i]][:,j]/fwd[i+1][j]
            bwd.append(bp)
        
        #Compute backward inference matrix
        for i in range(priormat.shape[0]):
            bp=np.zeros(len(fwd[-1]))
            bp[i]=1
            for j in bwd:
                bp=bp.reshape((1,-1)).dot(j)
            priormat[i]=bp
        self.bwdmat[ldim]=priormat
    
    
    def sample_pödkget(self,numex,rndpool,label,ldim):
        assert len(label)==numex
        while ldim<0:
            ldim+=self.edges.shape[0]+1
        if ldim not in self.bwdmat.keys():
            self.sample_pödkget_init(ldim)
        
        if rndpool is None:
            rndpool=np.random.rand((self.edges.shape[0]+1)*numex)
        
        gen=np.zeros((numex,self.edges.shape[0]+1),dtype=np.int64)
        gen[:,ldim]=label
        #tr1/=tr1.max()
        k=np.zeros(1,dtype=np.int64)
        for k in np.arange(numex):
            tr1=np.dot(np.hstack([0,self.bwdmat[ldim][label[k]]]).reshape([1,-1]),np.triu(np.ones([len(self.prior)+1,len(self.prior)+1]))).reshape(-1)
            gen[k,self.root]=np.where((tr1[:-1]<=rndpool[k])*(tr1[1:]>rndpool[k]))[0][0]
        rndpool=rndpool[numex:]
        for l in np.arange(self.edges.shape[0]):
            pöd=self.pödkget[l]
            tr=np.zeros((pöd.shape[0],pöd.shape[1]+1))
            for k in np.arange(pöd.shape[0]):
                tr[k,:]=np.dot(np.hstack([0,pöd[k,:].reshape(-1)]).reshape((1,-1)),
                           np.triu(np.ones([pöd.shape[1]+1,pöd.shape[1]+1]))).reshape(-1)
            for k in range(numex):
                if self.edges[l,1]==ldim:
                    continue
                x=gen[k,self.edges[l,0]].astype(int)
                tr2=tr[x,:]
                #tr2/=tr2.max()
                gen[k,self.edges[l,1]]=np.where((tr2[:-1]<=rndpool[l+1])*(tr2[1:]>rndpool[l+1]))[0][0]
            rndpool=rndpool[gen.shape[0]-1:]
        return gen,rndpool
        
                    






'''


disk='/media/bvegetas/亂數天下'
#disk='/media/bvegetas/Kekacauan'
#disk='/media/bvegetas/Penghinaan'
#disk='/media/bvegetas/Äptfen'
diskdata='/media/bvegetas/DJZenö'
hbräntsj=diskdata+'/Häekkä/liberal.csv'
hencoding='windows-1254'
htbl=np.loadtxt(hbräntsj,str,'#','\t',None,1,encoding=hencoding)
hseqn=htbl[:,0]
htti=htbl[:,1:]
hlengths=[]
for i in np.unique(hseqn):
    if len(np.where(hseqn==i)[0])>0:
        hlengths.append(len(np.where(hseqn==i)[0]))
        
import numpy as np
hbräntsj=diskdata+'/Häekkä/liberal.csv'
hencoding='windows-1254'
htbl=np.loadtxt(hbräntsj,str,'#','\t',None,1,encoding=hencoding)
hseqn=htbl[:,0]
htti=htbl[:,1:]
hlengths=[]
for i in np.unique(hseqn):
    if len(np.where(hseqn==i)[0])>0:
        hlengths.append(len(np.where(hseqn==i)[0]))

uc=duc()
uc.fit(htti)
data=uc.predict(htti)
tc=tancell()
tc.fit(data)
gen1,_=tc.sample_pödkget(data.shape[0], np.random.rand(65536), data[:,7], 7)
gen2,_=tc.sample_pödkget(data.shape[0], np.random.rand(65536), data[:,-1], -1)
gen,_=tc.sample(65536,np.random.rand(2000000))    
'''
#spanning(htti)
