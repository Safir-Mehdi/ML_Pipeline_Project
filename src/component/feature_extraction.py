import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('config/data_config/mappings.json', 'r') as json_file:
            self.mappings = json.load(json_file)
    
    def fit(self, X, y=None):
        # Nothing to fit, as this is a pure transformation
        return self
    
    def transform(self, X):
        # 1. Extract Age Groups from -> Age
        X['age_group'] = pd.cut(
            x=X['age'],
            bins=[0, 18, 35, 55, 100],
            labels=['Childern', 'Young Adults', 'Middle Aged', 'Seniors']
            ).astype('object')

        # 2. Obtain Employment Type from -> Hours Per Week
        X['employment_type'] = pd.cut(
            x=X['hours_per_week'],
            bins=[0, 20, 40, 91],
            labels=['Part-Time', 'Full-Time', 'Over-Time']
            ).astype('object')

        # 3. Get Work-Life Balance from -> Hours Per Week and
        X['work_life_balance'] = X['hours_per_week']/168

        # 4. Fetch Over Time Flag from -> Employment Type
        X['over_time_flag'] = np.where(
            X['employment_type'] == 'Over-Time', 1, 0
            )  # this is also by using .apply()

        # 5. Secure Net Capital from -> Capilat Gain & Capilat Loss
        X['net_capital'] = X['capital_gain'] - X['capital_loss']

        # 6. Gather Education Level Group from -> Education
        education_map = self.mappings['education_map']
        X['education_level_group'] = X['education'].map(education_map)

        # 7. Collect Is Educated Flage from -> Education Number
        X['is_educated_flag'] = X['education_num'].apply(lambda x: 1 if x >10 else 0)  # Educated if Education Level > 10 (Threshold)

        # 8. Coin Year of Education Remaining from -> Education Number
        X['year_of_education_remaining'] = X['education_num'].max() - X['education_num']

        # 9. Attain Is Married Flage from -> Marital Status
        is_married = X['marital_status'].str.contains(r'\bMarried\b', regex=True)
        X['is_married_flag'] = np.where(is_married, 1, 0)

        # 10. Extract Region from -> Native Country
        country_to_region = self.mappings['country_to_region_mapping']
        X['region'] = X['native_country'].map(country_to_region)
        
        # Returns -> Pandas DataFrame
        return X