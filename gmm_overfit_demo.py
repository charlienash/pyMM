# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:18:41 2016

@author: charlie
"""

import numpy as np
from models.pyMix import GMM, MPPCA, MFA
from sklearn.cross_validation import KFold

dat_path = '/tmp/virus3.dat'
data = np.loadtxt(dat_path)

# normalize data
X = (data - data.mean(axis=0)) / data.std(axis=0)


# Get cv splits
k = 10
n_examples = X.shape[0]
kf = KFold(n_examples, k)

# Fit model evaluate with cross-validation
n_components = 1
val_score = 0
for train, test in kf:       
    gmm = GMM(n_components=n_components)
    gmm.fit(X[train], init_method='kmeans')
    print('Val score: {}'.format(gmm.score(X[test])))
    val_score += gmm.score(X[test])
mean_val_score = val_score / k

