from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# TODO: Make my own standard scaler
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        return self.scaler.transform(X)
