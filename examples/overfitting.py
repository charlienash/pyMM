# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.cross_validation import train_test_split

from mixy.models import GMM, SphericalGMM, DiagonalGMM, MPPCA, MFA

# Load dataset
X = load_diabetes()['data']
X = X[:100]

# normalize
#X = (X - X.mean(axis=0)) / X.std(axis=0)

# Get train test split
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)

# Fit full GMM and MFA
n_components = 3
gmm = GMM(n_components=n_components, robust=True)
gmm.fit(X_train)
mfa = MPPCA(n_components=n_components, latent_dim=4, robust=True)
mfa.fit(X_train)

# Get test ll
print(gmm.score(X_train))
print(mfa.score(X_train))

print(gmm.score(X_test))
print(mfa.score(X_test))

# Plot first two dims
#X_plot = X_train[:,:2]
#plt.scatter(X_plot[:,0], X_plot[:,1])
#plt.hist(X_plot[:,0])