# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:37:59 2016

@author: charlie
"""

from pyMix import GMM, SphericalGMM, DiagonalGMM, MPPCA
from helper import _generate_mixture_data
import matplotlib.pyplot as plt

def main():
    n_examples = 1000
    data_dim = 2
    n_components = 2
    X = _generate_mixture_data(data_dim, n_components, n_examples)    
#    gmm = GMM(n_components=n_components)
#    gmm = SphericalGMM(n_components=n_components)
#    gmm = DiagonalGMM(n_components=n_components)
    gmm = MPPCA(n_components=n_components, latent_dim=1)
    
    # Fit GMM
    gmm.fit(X, init_method='kmeans')
    X_samples = gmm.sample(1000)
    
    # Plot results
    plt.clf()
    plt.scatter(X[:,0], X[:,1], color='b')
    #plt.hold(True)
    plt.scatter(X_samples[:,0], X_samples[:,1], color='r')
    plt.axis('equal')
    plt.show()
    
main()
