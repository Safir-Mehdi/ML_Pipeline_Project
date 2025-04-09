import os
import json
import joblib
import pandas as pd
from src.utils import fetch_data, transforme_DataFrame
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

class TargetFeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, training=True):
        
        logging.info(msg='<<<<Target Transformation has been Started>>>>')
        
        file_path = os.path.join('config', 'data_config', 'transform_features.json')
        with open(file_path, 'r') as file:
            self.trans_config = json.load(file)
        
        self.training = training
        
        if self.training:
            
            logging.info(msg='Transformation is in Training Mode')
            self.trans = None
        else:
            
            logging.info(msg='Transformation is in Surving Mode')
            file_path = os.path.join('artifacts', 'column_transformers', 'target_feature_transformer.pkl')
            try:
                with open(file_path, 'rb') as file:
                    self.trans = joblib.load(file)

            except FileNotFoundError:
                raise FileNotFoundError(f"File not found:\n{file_path}.\nPlease ensure the file exists.")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, save=False):
        
        if not self.training:
            raise RuntimeError('''
                               fit() - can not be called, because transformation is surving mode.\n
                               If you want to use this transformer in training mode, please call fit() with training=True.
                               ''')
        
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
            raise TypeError('Both X and y must be Pandas DataFrame.')
        
        self.trans = ColumnTransformer(
            transformers=[
                ('target_encoder', TargetEncoder(), self.trans_config['target_features'])
            ],
            remainder='passthrough'
        )
        
        self.trans.fit(X=X, y=y)
        
        logging.info(msg='Transformer has been Fiffed')
        
        if save:
            
            file_path = os.path.join('artifacts', 'column_transformers', 'target_feature_transformer.pkl')
            with open(file_path, 'wb') as file:
                joblib.dump(self.trans, file_path)
            
            logging.info(msg='Transformer is Saved - file_path: {}'.format(file_path))
    
    
    def trasnfrom(self, X: pd.DataFrame):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be Pandas DataFrame.')
        
        trans = self.trans.transform(X=X)
        
        logging.info(msg='Transformation is Completed')
        
        return trans



# Run this command to test this module: src/component/target_feature_tranformation.py
if __name__ == '__main__':
    
    X_train_featured = fetch_data(FILE_NAME='X_train_transformed.csv', DIRECTORY_NAME='featured')
    X_test_featured = fetch_data(FILE_NAME='X_test_transformed.csv', DIRECTORY_NAME='featured')
    X_train = fetch_data(FILE_NAME='X_train_simple_transformed.csv', DIRECTORY_NAME='processed')
    X_test = fetch_data(FILE_NAME='X_train_simple_transformed.csv', DIRECTORY_NAME='processed')
    y = fetch_data(FILE_NAME='y_train_transformed.csv', DIRECTORY_NAME='featured')
    
    logging.info(msg=f'y type: {type(y)}')
    
    
    # 1. Creating a Pickle File of a Transformed Simple Featured Set
    # ---------------------------------------------------------------
    # trans = TargetFeatureTransform()
    # trans.fit(X=X_train, y=y, save=True)
    # X_train_trans = trans.trasnfrom(X=X_train)
    # X_test_trans = trans.trasnfrom(X=X_test)
    
    
    # 2. Creating a Pickle File of a New Transformed Featured Set
    # ------------------------------------------------------------
    trans = TargetFeatureTransform()
    trans.fit(X=X_train_featured, y=y, save=True)
    X_train_trans = trans.trasnfrom(X=X_train_featured)
    X_test_trans = trans.trasnfrom(X=X_test_featured)
    
    logging.info(
        msg=f'X_train Type:{type(X_train_trans)} Shape: {X_train_trans.shape}\nSample:\n{X_train_trans[:5]}'
        )
    logging.info(
        msg=f'X_test Type:{type(X_test_trans)} Shape: {X_test_trans.shape}\nSample:\n{X_test_trans[:5]}'
        )
