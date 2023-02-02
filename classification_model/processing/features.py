from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupTransformImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, group, trans):
        if not isinstance(variables, list):
            raise ValueError('variable should be a list')

        self.variables = variables
        self.group = group
        self.trans = trans
       
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
          X[var] = X[var].fillna(X.groupby(self.group)[var].transform(self.trans))

        return X