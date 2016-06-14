import numpy as np
from pyMM import GMM, SphericalGMM, DiagonalGMM, MPPCA, MFA
from util import _generate_mixture_data, plot_density, _gen_low_rank_data


def main():
    n_examples = 2000
    data_dim = 1000
    n_components = 1
    rank = 50
#    X = _generate_mixture_data(data_dim, n_components, n_examples)
    X = _gen_low_rank_data(data_dim, rank, n_examples)

    # Obscure data
    r = np.random.rand(n_examples, data_dim)
    X_miss = X.copy()
    X_miss[r > 0.6] = np.nan

    # Initialize model
#    gmm = GMM(n_components=n_components)
#    gmm = SphericalGMM(n_components=n_components, robust=True)
#    gmm = DiagonalGMM(n_components=n_components)
#    gmm = MPPCA(n_components=n_components, latent_dim=1)
    gmm = MFA(n_components=n_components, latent_dim=40, tol=1e-3)

    # Fit GMM
    gmm.stepwise_fit(X_miss, init_method='kmeans', batch_size=200,
                     step_alpha=0.7)
#    gmm.fit(X, init_method='kmeans')
    gmm.fit(X_miss, init_method='kmeans')

    # Plot results
#    plot_density(gmm, X=X, n_grid=50)

main()
