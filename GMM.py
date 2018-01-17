# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:31:19 2018

@author: lanlandetian
"""

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

    
def loadData():
    mean1 = [0,0]
    cov1 = [[1,0.5], [0.5,1]]
    
    mean2 = [5,5]
    cov2 = [[1,0], [0,1]]
    
    mean3 = [-5,1]
    cov3 = [[5,0], [0,5]]
    
    X = np.random.multivariate_normal(mean1,cov1,100)
    a = np.random.multivariate_normal(mean2,cov2,100)
    b = np.random.multivariate_normal(mean3,cov3,100)
    X = np.concatenate((X,a,b),axis = 0)
    
    return X

#多元高斯函数
def gaussian(x,mu,sigma):
    alpha = 0.1
    d = len(x)
    n = np.shape(sigma)[0]
    ex = - 1/2 * (np.mat(x - mu) * np.mat(linalg.inv( sigma + alpha * np.eye(n,n) )) \
    * np.mat(x - mu).T)[0,0]
    determinant = linalg.det(sigma)
    ret = 1 / ( np.power((2*np.pi),d/2) * np.power(determinant, 1/2) ) * np.exp(ex)
    return ret

#目标函数
def Loss(X,k,pi,mu,sigma):
    [m,n] = np.shape(X)
    W = np.zeros((m,k))
    #求解W
    for i in range(0,m):
        total = 0
        for j in range(0,k):
            total += pi[j]*gaussian(X[i],mu[j],sigma[j])
            
        for j in range(0,k):
            W[i,j] = pi[j]*gaussian(X[i],mu[j],sigma[j]) / total
    
    ret = 0
    for i in range(0,m):
        for j in range(0,k):
            ret += W[i,j]* np.log(pi[j]*gaussian(X[i],mu[j],sigma[j]) / W[i,j])
            
    return ret
    
#EM算法主函数
def EM(X,k = 3,iter = 50):
    [m,n] = np.shape(X)
    #初始化参数
    pi = np.array([0.1,0.5,0.4])
    mu = np.array([[0,0],
                   [1,5],
                   [-5,3]])
    sigma = np.zeros((k,n,n))
    sigma[0] = np.array([[1,0.5],[0.5,1]])
    sigma[1] = np.array([[3,0],[0,3]])
    sigma[2] = np.array([[2,0],[0,2]])
    
    oldLoss = Loss(X,k,pi,mu,sigma)
    J = []
    #主循环
    W = np.zeros((m,k))
    for step in range(0,iter):
        #求解W
        for i in range(0,m):
            total = 0
            for j in range(0,k):
                total += pi[j]*gaussian(X[i],mu[j],sigma[j])
            for j in range(0,k):
                W[i,j] = pi[j]*gaussian(X[i],mu[j],sigma[j]) / total
       
        #更新pi，mu,sigma
        newPi = np.zeros(k)
        newMu = np.zeros((k,n))
        newSigma = np.zeros((k,n,n))
        for j in range(0,k):
            tmpW = 0
            for i in range(1,m):
                tmpW += W[i,j]
                newMu[j] += W[i,j] * X[i]
                newSigma[j] += W[i,j] * np.array(np.mat(X[i] - mu[j]).T * np.mat(X[i] - mu[j]))
            newPi[j] = 1 / m * tmpW
            newMu[j] /= tmpW
            newSigma[j] /= tmpW
          
        #判断是否满足终止条件
        curLoss =  Loss(X,k,newPi,newMu,newSigma)
        if np.abs(oldLoss - curLoss) < 0.1:
            break
        else:
            print(oldLoss)
            
            J.append(oldLoss)
            oldLoss = curLoss
            pi = newPi
            mu = newMu
            sigma = newSigma
    plt.figure()
    plt.plot(J,'-*')
    plt.title('Loss Function')
    return pi,mu,sigma,W
        
if __name__ == '__main__':
    X = loadData()
    
    plt.figure()
    plt.plot(X[:,0],X[:,1],'*')
    plt.axis('equal')
    plt.title('original data')
    #运行EM算法
    k = 3
    [m,n] = np.shape(X)
    pi,mu,sigma,W = EM(X,k)
    
    clusters = dict()
    for i in range(0,k):
        clusters[i] = []

    for i in range(0,m):
        index = np.argmax(W[i])
        clusters[index].append(i)
    
    plt.figure()
    plt.plot(X[clusters[0],0], X[clusters[0],1],'*')
    plt.plot(X[clusters[1],0], X[clusters[1],1],'o')
    plt.plot(X[clusters[2],0], X[clusters[2],1],'d')
    plt.title('result')





