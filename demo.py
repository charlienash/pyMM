import numpy as np
from pyMM import GMM, SphericalGMM, DiagonalGMM, MPPCA, MFA
from util import _generate_mixture_data, plot_density


def main():
    n_examples = 2500
    data_dim = 500
    n_components = 1
    X = _generate_mixture_data(data_dim, n_components, n_examples)

    # Obscure data
    r = np.random.rand(n_examples, data_dim)
    X_miss = X.copy()
    X_miss[r > 0.7] = np.nan

    # Initialize model
    gmm = GMM(n_components=n_components, robust=True, SMALL=1e-5)
#    gmm = SphericalGMM(n_components=n_components, robust=True)
#    gmm = DiagonalGMM(n_components=n_components)
#    gmm = MPPCA(n_components=n_components, latent_dim=1)
#    gmm = MFA(n_components=n_components, latent_dim=50)

    # Fit GMM
#    gmm.stepwise_fit(X, init_method='kmeans')
    gmm.fit(X, init_method='kmeans')
#    gmm.fit(X_miss, init_method='kmeans')

    # Plot results
#    plot_density(gmm, X=X, n_grid=50)

main()
