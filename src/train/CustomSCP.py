from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CustomSCP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.n_channels = 64
        self.filters_ = None

    def fit(self, X, y):
        self.filters_ = self.compute_scp(X, y)
        return self

    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("You must fit the transformer before transforming data.")
        n_epochs, n_channels_times = X.shape
        n_times = n_channels_times // self.n_channels
        X = X.reshape(n_epochs, self.n_channels, n_times)

        X_filtered = np.array([self.filters_ @ epoch for epoch in X])

        variance = np.var(X_filtered, axis=2) + 1e-10
        X_features = np.log(variance)

        return X_features

    def compute_scp(self, X, y):
        n_epochs, n_channels_times = X.shape
        n_times = n_channels_times // self.n_channels
        X = X.reshape(n_epochs, self.n_channels, n_times)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('CSP implementation only supports binary classification')

        class_1, class_2 = classes[0], classes[1]
        X_class1 = X[y == class_1]
        X_class2 = X[y == class_2]

        covs_class1 = self.custom_covariance_matrix(X_class1)
        covs_class2 = self.custom_covariance_matrix(X=X_class2)

        cov_class1 = np.mean(covs_class1, axis=0)
        cov_class2 = np.mean(covs_class2, axis=0)

        composite_cov = cov_class1 + cov_class2

        eigenvalues, eigenvectors = self.custom_eigen_decomposition(composite_cov)

        epsilon = 1e-10
        eigenvalues = np.where(eigenvalues < epsilon, epsilon, eigenvalues)

        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        S0 = whitening_matrix @ cov_class1 @ whitening_matrix.T

        eigenvalues_S0, eigenvectors_S0 = self.custom_eigen_decomposition(S0)

        sorted_indices = np.argsort(eigenvalues_S0)[::-1]
        eigen_vectors_S0 = eigenvectors_S0[:, sorted_indices]

        filters = eigen_vectors_S0.T @ whitening_matrix

        n_filters = self.n_components // 2
        selected_filters = np.vstack([filters[:n_filters], filters[-n_filters:]])

        return selected_filters

    @staticmethod
    def custom_covariance_matrix(X):
        n_epochs, n_channels, n_times = X.shape
        cov_matrices = []
        for epoch in X:
            epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
            cov = (epoch @ epoch.T) / (n_times - 1)
            cov /= np.trace(cov)
            cov_matrices.append(cov)
        return cov_matrices

    @staticmethod
    def custom_eigen_decomposition(matrix):
        n = matrix.shape[0]
        eigenvalues = np.zeros(n)
        eigenvectors = np.zeros((n, n))

        residual_matrix = matrix.copy()

        for i in range(n):
            b_k = np.random.rand(n)
            for _ in range(100):
                b_k1 = residual_matrix @ b_k
                b_k1_norm = np.linalg.norm(b_k1)
                b_k = b_k1 / b_k1_norm
            eigenvalue = b_k.T @ residual_matrix @ b_k
            eigenvalues[i] = eigenvalue
            eigenvectors[:, i] = b_k
            residual_matrix = residual_matrix - eigenvalue * np.outer(b_k, b_k)
        return eigenvalues, eigenvectors
