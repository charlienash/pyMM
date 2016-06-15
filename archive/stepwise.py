    def stepwise_fit(self, X, params_init=None, init_method='kmeans',
                     em_type='standard', batch_size=250, step_alpha=0.7):
        """ Fit the model using EM with data X.

        Args
        ----
        X : array, [nExamples, nFeatures]
            Matrix of training data, where nExamples is the number of
            examples and nFeatures is the number of features.
        """
        if np.isnan(X).any():
            self.missing_data = True
        else:
            self.missing_data = False

        # Check for missing values and remove if whole row is missing
        X = X[~np.isnan(X).all(axis=1), :]
        n_examples, data_dim = np.shape(X)
        self.data_dim = data_dim
        self.n_examples = n_examples

        if params_init is None:
            params = self._init_params(X, init_method)
        else:
            params = params_init

        # Get batch indices
        if em_type == 'standard':
            batch_size = n_examples
        elif em_type == 'stepwise':
            batch_size = batch_size
        batch_id = np.hstack([np.arange(0, n_examples, batch_size),
                              n_examples])
        n_batch = batch_id.size - 1

        # Do stepwise EM
        oldL = -np.inf
        k = 0
        for i in range(self.max_iter):

            # Do full E-Step if stepwise EM
            if i == 0:
                ss, sample_ll = self._e_step(X, params)

            # Evaluate likelihood
            ll = sample_ll.mean() / self.data_dim
            if self.verbose:
                print("Iter {:d}   NLL: {:.4f}   Change: {:.4f}".format(i,
                      -ll, -(ll-oldL)), flush=True)

            # Break if change in likelihood is small
            if np.abs(ll - oldL) < self.tol:
                break
            oldL = ll

            for b in range(n_batch):

                # M-Step
                params = self._m_step(ss, params)

                # Get batch
                X_batch = X[batch_id[b]:batch_id[b+1]]
                batch_size_current = X_batch.shape[0]

                # Do batch e-Step
                batch_ss, sample_ll = self._e_step(X_batch, params)

                # Apply stepwise update to summary stats
                if em_type == 'stepwise':
                    step = (k + 2)**(-step_alpha)
                    for stat in ss:
                        current_list = ss[stat]
                        batch_list = batch_ss[stat]
                        ss[stat] = [(1 - step) * batch_list[k] /
                                    batch_size_current*n_examples +
                                    step * current_list[k] for k in
                                    range(self.n_components)]

                    # Increment k
                    k += 1

                # Apply standard update to summary stats
                elif em_type == 'standard':
                    ss = batch_ss

        else:
            if self.verbose:
                print("PPCA did not converge within the specified" +
                      " tolerance. You might want to increase the number of" +
                      " iterations.")

        # Update Object attributes
        self.params = params
        self.trainNll = ll
        self.isFitted = True