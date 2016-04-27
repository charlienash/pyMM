import numpy as np
#import matplotlib.pyplot as plt

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
    
def _get_rand_cov_mat(dim):
    Sigma = np.random.randn(dim, dim)
    Sigma = Sigma.dot(Sigma.T)
    Sigma = Sigma + dim*np.eye(dim)
    return Sigma
    
def _generate_mixture_data(dim, n_components, n_samples):
    components = np.random.rand(n_components)
    components = components / np.sum(components)
    Sigma_list =  [_get_rand_cov_mat(dim) for j in range(n_components)]
    mu_list = [8*np.random.randn(dim) for j in range(n_components)]
    components_cumsum = np.cumsum(components)
    samples = np.zeros([n_samples, dim])
    for n in range(n_samples):
        r = np.random.rand(1)
        z = np.argmin(r > components_cumsum)               
        samples[n] = np.random.multivariate_normal(mu_list[z], Sigma_list[z])  
    return samples
#    
#X = _generate_mixture_data(2, 2, 1000)    
#plt.clf()
#plt.scatter(X[:,0], X[:,1])
#def _gmm_log_lik(X, mu_list, Sigma_list, components):