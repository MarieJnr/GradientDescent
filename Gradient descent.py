# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:14:50 2022

@author: 22541
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#lecture des données 
data = np.genfromtxt("dataset.csv",delimiter=";")
X = data[:,:2]
Y = data[:,2]
print(X[10])
#Representation graphique 
plt.scatter(X[:15,1],Y[:15])
plt.grid()
#Modele
def h(theta,x):
    #return theta[0]*x[0] + theta[1]*x[1] + theta[2]*x[2]
    return np.dot(theta,x)
v1 = np.array([1,1])
v2 = np.array([1,-1])
print(h(v1,v2))
v=X[0]
print('v= ',v)
theta = np.array([1,-1])
print('h= ',h(theta,v))
#Fonction gradient
def grad(theta,N,j):
    g = 0
    for i in range(N):
        x = X[i]
        y = Y[i]
        g = g + (h(theta,x) - y)*x[j]
    return g/N
#Fonction cout 
def J(theta,N):
    er=0
    for i in range(N):
        x = X[i]
        y = Y[i]
        er =er +(np.dot(theta,x) - y)**2
    return er/(2*N)
#Calcul de parametre du modele 
start_time = time.time()
N = 50
l = 0.002 #learning rate
theta = np.array([1,1])
t0 = theta[0]
t1 = theta[1]
cost = []
epoch = []

for k in range(30):
    
    epoch.append(k)
    gd = grad(theta,N,0)
    t0 = t0 -l*gd
    gd = grad(theta,N,1)
    t1 = t1 -l*gd
    
    theta = np.array([t0,t1])
    cost.append(J(theta,N))
    print('epoch===: ',k, 'Err: ',cost[k])
end_time = time.time()
plt.plot(epoch,cost)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('cost')

print('satart=',start_time)
print('endtime = ',end_time)
print('Temps execution = ', end_time-start_time, 's')
print('t0 = ',t0)
print('t1 = ',t1)

