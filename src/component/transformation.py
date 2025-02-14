import numpy as np
import pandas as pd
from typing import Union
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin

class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = None

    def set_feature_names(self, feature_names):
        """
        Set feature names when input is in numpy array format.
        """
        self.feature_names = feature_names

    def get_feature_names_out(self, input_features: Union[str, list, tuple] = None):
        """
        Return the feature names after the transformation.
        """
        if input_features is not None:
            return np.array(input_features, dtype='object')
        elif self.feature_names is not None:
            return np.array(self.feature_names, dtype='object')
        else:
            raise ValueError('Feature names are not available. Ensure the transformer is fitted and feature names are set.')

class FrequencyEncoder(BaseTransformer):
    def __init__(self, fillna = -1):
        super().__init__()
        # Initialize the FrequencyEncoder with an empty frequency map.
        self.frequency_map = {}
        self.fillna = fillna
        # self.feature_names = None
        
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError('Input data must be a pandas DataFrame or numpy array.')
        self.frequency_map = X.apply(lambda feature: feature.value_counts(normalize=True).to_dict(), axis=0)
        return self
    
    def transform(self, X):
        # return X.apply(lambda feature: feature.map(self.frequency_map[feature.name]))
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError('Input data must be a pandas DataFrame or numpy array.')
        return X.apply(lambda feature: feature.map(self.frequency_map[feature.name]).fillna(self.fillna))
           
    
    # This class is not compatible with the ColumnTransformer in Scikit-learn, So we need some changes in it
class Winsorizer(BaseTransformer):
    def __init__(self, feature_limits: dict = None):
        super().__init__()
        self.feature_limits = feature_limits
        # self.feature_names = None
    
    def fit(self, X, y=None):
        # print(self.feature_limits)
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError('Input data must be a pandas DataFrame or numpy array.')
        for feature, limit in self.feature_limits.items():
            if feature in X.columns:
                lower_percentile, upper_percentile = limit
                # print(f'Winsorizing feature {feature} with limits: {lower_percentile} and {upper_percentile}')
                X[feature] = winsorize(X[feature], limits=(lower_percentile, 1 - upper_percentile))
            else:
                raise ValueError(f'Feature {feature} not found in the input data.')
        return X