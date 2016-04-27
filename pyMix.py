"""Probabilistic principal components analysis (PPCA)

A generative latent linear variable model.

PPCA assumes that the observed data is generated by linearly transforming a
number of latent variables and then adding spherical Gaussian noise. The
latent variables are drawn from a standard Gaussian distribution.

This implementation is based on David Barber's Matlab implementation:
http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Main.Software

This implementation uses the EM algorithm to handle missing data.
"""

# Author: Charlie Nash <charlie.tc.nash@gmail.com>

import numpy as np
import numpy.random as rd
#from numba import jit
from random import seed
#import GenModel
from helper import _mv_gaussian_pdf, _get_rand_cov_mat
#from scipy.stats import multivariate_normal

class GMM():
    """Probabilistic principal components analysis (PPCA).

    A generative latent linear variable model.

    PPCA assumes that the observed data is generated by linearly transforming a
    number of latent variables and then adding spherical Gaussian noise. The
    latent variables are drawn from a standard Gaussian distribution.

    The parameters of the model are the transformation matrix (principal
    components) the mean, and the noise variance.

    PPCA performs maximum likelihood or MAP estimation of the model parameters using
    the expectation-maximisation algorithm (EM).

    Attributes
    ----------

    latentDim : int
        Dimensionality of latent space. The number of variables that are
        transformed by the principal components to the data space.

    components : array, [latentDim, nFeatures]
        Transformation matrix parameter.

    bias: array, [nFeatures]
        Bias parameter.

    noiseVariance : float
        Noise variance parameter. Variance of noise that is added to linearly
        transformed latent variables to generate data.

    standardize : bool, optional
        When True, the mean is subtracted from the data, and each feature is
        divided by it's standard deviation so that the mean and variance of
        the transformed features are 0 and 1 respectively.

    componentPrior : float >= 0
        Gaussian component matrix hyperparameter. If > 0 then a Gaussian prior
        is applied to each column of the component matrix with covariance
        componentPrior^-1 * noiseVariance. This has the effect
        of regularising the component matrix.

    tol : float
        Stopping tolerance for EM algorithm

    maxIter : int
        Maximum number of iterations for EM algorithm

    Notes
    -----

    TODO

    Examples
    --------

    TODO
    """
    def __init__(self, n_components, tol=1e-3, max_iter=1000, random_state=0, 
                  verbose=True):
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.isFitted = False
        self.verbose = verbose
        self.n_components = n_components

    def _e_step(self, X, params):
        """ E-Step of the EM-algorithm.

        The E-step takes the existing parameters, for the components, bias
        and noise variance and computes sufficient statistics for the M-Step
        by taking the expectation of latent variables conditional on the
        visible variables. Also returns the likelihood for the data X and
        projections into latent space of the data.

        Args
        ----
        X : array, [nExamples, nFeatures]
            Matrix of training data, where nExamples is the number of
            examples and nFeatures is the number of features.
        W : array, [dataDim, latentDim]
            Component matrix data. Maps latent points to data space.
        b : array, [dataDim,]
            Data bias.
        sigmaSq : float
            Noise variance parameter.

        Returns
        -------
        ss : dict

        proj :

        ll :
        """
        # Get params
        Sigma_list = params['Sigma_list']
        mu_list = params['mu_list']
        components = params['components']

        n_examples, data_dim = X.shape
        
        # Compute responsibilities
        r = np.zeros([n_examples, self.n_components])
        for k, mu, Sigma in zip(range(self.n_components), mu_list, Sigma_list):
#            r[:,k] = multivariate_normal.pdf(X, mu, Sigma)
            r[:,k] = _mv_gaussian_pdf(X, mu, Sigma)
        r = r * components
        r_sum = r.sum(axis=1)
        responsibilities = r / r_sum[:,np.newaxis]
            
        # Store sufficient statistics in dictionary
        ss = {
            'responsibilities' : responsibilities
             }
             
        # Compute log-likelihood
        ll = np.log(r_sum).sum()

        return ss, ll

    def _m_step(self, X, ss, params):
        """ M-Step of the EM-algorithm.

        The M-step takes the sufficient statistics computed in the E-step, and
        maximizes the expected complete data log-likelihood with respect to the
        parameters.

        Args
        ----
        ss : dict

        Returns
        -------
        params : dict

        """
        resp = ss['responsibilities']
#        resp_sum = 
        
        # Update components param
        components = np.mean(resp, axis=0)

        # Update mean / Sigma params
        mu_list = []
        Sigma_list = []
        for r in resp.T:
            mu = np.sum(X*r[:,np.newaxis], axis=0) / r.sum()
            mu_list.append(mu)          
            Sigma = (X*r[:,np.newaxis]).T.dot(X) / r.sum() - np.outer(mu, mu)
            Sigma_list.append(Sigma)
            
        # Store params in dictionary
        params = {
            'Sigma_list' : Sigma_list,
            'mu_list' : mu_list,
            'components' : components
             }

        return params


    def fit(self, X, paramsInit=None):
        """ Fit the model using EM with data X.

        Args
        ----
        X : array, [nExamples, nFeatures]
            Matrix of training data, where nExamples is the number of
            examples and nFeatures is the number of features.
        """
        n_examples, data_dim = np.shape(X)
        n_components = self.n_components

        if paramsInit is None:
            seed(self.random_state)
            params = {
                      'Sigma_list' :  [_get_rand_cov_mat(data_dim) for j in range(n_components)],
                      'mu_list' : [np.random.randn(data_dim) for j in range(n_components)],
                      'components' : 1/n_components * np.ones(n_components)
            }
        else:
            params = paramsInit

        oldL = -np.inf
        for i in range(self.max_iter):

            # E-Step
            ss, ll = self._e_step(X, params)

            # Evaluate likelihood
            if self.verbose:
                print("Iter {:d}   NLL: {:.3f}   Change: {:.3f}".format(i,
                      -ll, -(ll-oldL)), flush=True)

            # Break if change in likelihood is small
            if np.abs(ll - oldL) < self.tol:
                break
            oldL = ll

            # M-step
            params = self._m_step(X, ss, params)

        else:
            if self.verbose:
                print("PPCA did not converge within the specified tolerance." +
                      " You might want to increase the number of iterations.")

        # Update Object attributes
        self.components = params['components']
        self.Sigma_list = params['Sigma_list']
        self.mu_list= params['mu_list']
        self.trainNll = ll
        self.isFitted = True
        self.data_dim = data_dim

    def sample(self, n_samples=1):
        """Sample from fitted model.

        Sample from fitted model by first sampling from latent space
        (spherical Gaussian) then transforming into data space using learned
        parameters. Noise can then be added optionally.

        Parameters
        ----------
        nSamples : int
            Number of samples to generate
        noisy : bool
            Option to add noise to samples (default = True)

        Returns
        -------
        dataSamples : array [nSamples, dataDim]
            Collection of samples in data space.
        """
        if  not self.isFitted:
            print("Model is not yet fitted. First use fit to learn the model"
                   + " params.")
        else:
            components_cumsum = np.cumsum(self.components)
            samples = np.zeros([n_samples, self.data_dim])
            for n in range(n_samples):
                r = np.random.rand(1)
                z = np.argmin(r > components_cumsum)               
                samples[n] = rd.multivariate_normal(self.mu_list[z], 
                    self.Sigma_list[z])                
            return samples


    def score(self, X):
        """Compute the average log-likelihood of data matrix X

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            The data

        Returns
        -------
        meanLl: array, shape (n_samples,)
            Log-likelihood of each sample under the current model
        """
        if not self.isFitted:
            print("Model is not yet fitted. First use fit to learn the model"
                   + " params.")
        else:
            # Get fitted parameters
            params = {
                     'sigmaSq': self.noiseVariance,
                     'W' : self.components,
                     'b' : self.bias
                     }

            # Apply one step of E-step to get the total log likelihood
            L = self._e_step(X, params)[1]

            # Divide by number of examples to get average log likelihood
            n_examples = np.shape(X)[0]
            mean_ll = L / n_examples
            return mean_ll
            


