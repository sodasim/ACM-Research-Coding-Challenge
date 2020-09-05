# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:49:09 2020

@author: sodasim
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt

sourceData = genfromtxt('ClusterPlot.csv', delimiter=",")
sourceData = sourceData[1:]
data = [0] * len(sourceData)
for i in range(len(sourceData)):
    data[i] = sourceData[i][1:]
    
data = StandardScaler().fit_transform(data)

db = DBSCAN(eps=0.3, min_samples=10).fit(data)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters)