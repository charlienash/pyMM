# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:37:59 2016

@author: charlie
"""

from pyMix import GMM, SphericalGMM, DiagonalGMM, MPPCA
from util import _generate_mixture_data, plot_cov_ellipse, plot_density
import matplotlib.pyplot as plt
#import numpy as np

def main():
    n_examples = 1000
    data_dim = 2
    n_components = 3
    X = _generate_mixture_data(data_dim, n_components, n_examples)    
#    gmm = GMM(n_components=n_components)
#    gmm = SphericalGMM(n_components=n_components)
#    gmm = DiagonalGMM(n_components=n_components)
    gmm = MPPCA(n_components=n_components, latent_dim=1)
    
    # Fit GMM
    gmm.fit(X, init_method='kmeans')
    
#    # Get covariance matrices and means
#    cov_list = gmm._params_to_Sigma(gmm.params)
#    mu_list = gmm.params['mu_list']
    
    # Plot results
    plot_density(gmm, X=X)
    plt.savefig('test.pdf', dpi=600)
           
main()
