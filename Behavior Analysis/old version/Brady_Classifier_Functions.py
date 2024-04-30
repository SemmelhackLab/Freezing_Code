import pandas as pd
import os, glob,sys, collections, itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold,KFold
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.spatial import distance
from scipy.stats import mode
from scipy.spatial.distance import squareform

class KnnDtw(object):
    def __init__(self, n_neighbors=7, max_warping_window=40, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        self.x = x
        self.l = l
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        return cost[-1, -1]
    
    def _dist_matrix(self, x, y):
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            p = ProgressBar(shape(dm)[0])
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
                    if dm_count%100==0:
                        p.animate(dm_count)
            
            dm = squareform(dm)
            return dm
        
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            dm_size = x_s[0]*y_s[0]
            
            p = ProgressBar(dm_size)
        
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
        
            return dm
        
    def predict(self, x):
        dm = self._dist_matrix(x, self.x)

        knn_idx = dm.argsort()[:, :self.n_neighbors]

        knn_labels = self.l[knn_idx]
        
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

def bradycardia_detection(input_trace_array,training_X_dir,training_Y_dir,n_neighbors0=7, max_warping_window0=40):
    X10 = np.load(training_X_dir)
    Y10 = np.load(training_Y_dir)
    
    clf = KnnDtw(n_neighbors=n_neighbors0, max_warping_window=max_warping_window0)
    clf.fit(X10, Y10)
    
    trace_list = []
    
    for t in range(0,input_trace_array.shape[0]):
        
        # normalize the heart rate trace
        temp_trace = input_trace_array[t,:]/np.mean(input_trace_array[t,300:899])
        trace_list.append(temp_trace)
    # only used 8-12s of the trace
    npy_trace_list = np.array(trace_list)[:,800:1200]
    
    dectection_result, proba = clf.predict(npy_trace_list)

    return dectection_result