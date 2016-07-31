""" Utilities functions for pyMM"""
import numpy as np


def _mv_gaussian_pdf(X, mu, Sigma):
    """
    Get Gaussian probability density for given data points and parameters.
    """
    return np.exp(_mv_gaussian_log_pdf(X, mu, Sigma))


def _mv_gaussian_log_pdf(X, mu, Sigma):
    """
    Get Gaussian log probability density for given data points and
        parameters.
    """
    d = mu.size
    dev = X - mu
    Sigma_inv = np.linalg.inv(Sigma)
    log_det = np.log(np.linalg.det(Sigma))
    maha = np.diag(dev.dot(Sigma_inv).dot(dev.T))
    return -0.5 * (d*np.log(2*np.pi) + log_det + maha)
