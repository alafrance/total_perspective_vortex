from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# TODO: Make my own classifier LDA
class CustomLDA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler.transform(X)
