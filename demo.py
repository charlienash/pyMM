import numpy as np
from pyMM import GMM, SphericalGMM, DiagonalGMM, MPPCA, MFA
from util import _generate_mixture_data, plot_density


def main():
    n_examples = 1000
    data_dim = 200
    n_components = 3
    X = _generate_mixture_data(data_dim, n_components, n_examples)

    # Obscure data
    r = np.random.rand(n_examples, data_dim)
    X_miss = X.copy()
    X_miss[r > 0.7] = np.nan

    # Initialize model
#    gmm = GMM(n_components=8)
#    gmm = SphericalGMM(n_components=n_components)
#    gmm = DiagonalGMM(n_components=n_components)
#    gmm = MPPCA(n_components=n_components, latent_dim=1)
    gmm = MFA(n_components=1, latent_dim=100, robust=False, SMALL=1e-8,
              tol=1e-3)
#    gmm = MPPCA_Miss(n_components=n_components, latent_dim=2)
#    gmm = MFA_Miss(n_components=n_components, latent_dim=1)

    # Fit GMM
    gmm.fit(X_miss, init_method='kmeans')
#    gmm.fit(X, init_method='kmeans')

#    print(gmm.score_samples(X))

    # Plot results
#    plot_density(gmm, X=X, n_grid=50)
#    plt.savefig('test.png', dpi=600)

main()
