import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
    
def plot_density(model, x_range='auto', y_range='auto', n_grid=100, 
                   with_scatter=True, X=None, contour_options=None, 
                   scatter_options=None, with_missing=False, X_miss=None):
                       
    # Set default options
    if contour_options is None:
        contour_options = {'cmap' : plt.cm.plasma}
    if scatter_options is None: 
        scatter_options = {'color' : 'w', 'alpha' : 0.5, 'lw' : 0}
        scatter_options_miss = {'color' : 'r', 'alpha' : 0.5, 'lw' : 0}
        
    # Automatic x_range and y_range
    if x_range == 'auto' and X is not None:
        x_range = [X[:,0].min() - 2, X[:,0].max() + 2]
    if y_range == 'auto' and X is not None:
        y_range = [X[:,1].min() - 2, X[:,1].max() + 2]
    
    # Setup grid for contour plot
    x_vec = np.linspace(x_range[0], x_range[1], n_grid)
    y_vec = np.linspace(y_range[0], y_range[1], n_grid) 
    x, y, = np.meshgrid(x_vec, y_vec)     
    X_grid = np.zeros([n_grid**2, 2])                        
    X_grid[:,0] = x.reshape(n_grid**2)
    X_grid[:,1] = y.reshape(n_grid**2)
    
    # Get sample log-likelihood from model
    grid_ll = model.score_samples(X_grid)
    grid_prob = np.exp(grid_ll) # Convert to probability density
    grid_prob = grid_prob.reshape((n_grid, n_grid))
        
    # Plot contours
    plt.contourf(x, y, grid_prob, **contour_options)

    # Add scatter if enables
    if with_scatter:
        if with_missing:
            id_miss = np.isnan(X_miss).any(axis=1)
            plt.scatter(X[~id_miss,0], X[~id_miss,1], **scatter_options)
            plt.scatter(X[id_miss,0], X[id_miss,1], **scatter_options_miss)
        else:
            plt.scatter(X[:,0], X[:,1], **scatter_options) # data                  

    # Plot options
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xticks([])
    plt.yticks([])
    plt.show()    
#    fig = plt.gcf()
#    fig.set_size_inches(8, 8, forward=True)
