# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:37:59 2016

@author: charlie
"""

from pyMix import GMM
from helper import _generate_mixture_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#%%
n_examples = 2000
data_dim = 2
n_components = 2
X = _generate_mixture_data(data_dim, n_components, n_examples)    
gmm = GMM(n_components=n_components)

# Initialize GMM with k_means
kmeans = KMeans(n_components)
kmeans.fit(X)

# Get params
mu_list = [k for k in kmeans.cluster_centers_]
Sigma_list = [np.cov(X[kmeans.labels_==k,:].T) for k in range(n_components)]
components = np.array([np.sum(kmeans.labels_==k) / n_examples for k in range(n_components)])
params_init = {
                'mu_list' : mu_list,
                'Sigma_list' : Sigma_list,
                'components' : components
                }
                
# Fit GMM
gmm.fit(X, params_init)
X_samples = gmm.sample(1000)

# Plot results
plt.clf()
plt.scatter(X[:,0], X[:,1], color='b')
plt.hold(True)
plt.scatter(X_samples[:,0], X_samples[:,1], color='r')
plt.show()