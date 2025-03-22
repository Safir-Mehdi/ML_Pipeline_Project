import joblib
import json
import pandas as pd
import numpy as np
from typing import Union, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import logging
from src.utils import transforme_DataFrame, fetch_data
from src.component.feature_transformation import FeatureTransformation
from src.component.feature_extraction import FeatureExtractor

class TargetTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, feature_transformer: FeatureTransformation):
        
        logging.info(msg='***** Target Transformation Started*****')
        
        # Check if there is a valid feature transformer
        if not isinstance(feature_transformer, FeatureTransformation):
            raise TypeError(f"Please provide FeatureTransformation obj - got {type(feature_transformer)}")
        
        # Initialize y train & test form feature_fransformer
        self.y_train = feature_transformer.y_train
        self.y_test = feature_transformer.y_test
        
        # Initialize label features from feature transformer
        self.label_features = feature_transformer.transform_features['label_features']
        
        # Initialize label encoder
        self.encoder = LabelEncoder()
    
    def fit(self, save: bool = False):
        
        logging.info('>>> Fitting Target Feature <<<')
        
        # Fit label encoder
        self.encoder.fit(self.y_train)
        
        # Save label encoder if required
        if save:
            with open('artifacts/label_encoder/label_encoder.pkl', 'wb') as file:
                joblib.dump(self.encoder, file)
                
                logging.info(msg='Label Encoder has been Saved')
        
        logging.info(msg='>>> Target Feature Fitted Successfully <<<')
        
        # Return <self>
        return self
    
    def transform(self, X=pd.DataFrame, save: bool = False) -> Tuple[pd.Series, pd.Series]:
        
        logging.info(msg='>>> Transforming Target Feature <<<')
        
        # Transform y_train & y_test using label encoder
        y_train_transformed = self.encoder.transform(y=self.y_train)
        y_test_transformed = self.encoder.transform(y=self.y_test)
        
        # Converte numpy array into pandas Series
        y_train_transformed = transforme_DataFrame(y_train_transformed, self.encoder)
        y_test_transformed = transforme_DataFrame(y_test_transformed, self.encoder)
        
        # Assigning names to pandas Series
        y_train_transformed.name = self.y_train.name
        y_test_transformed.name = self.y_test.name
        
        # Save transformed y_train & y_test if required
        if save:
            y_train_transformed.to_frame.to_csv('data/target_feature/y_train_transformed.csv', index=False)
            y_test_transformed.to_frame.to_csv('data/target_feature/y_test_trasnformed.csv', index=False)
            
            logging.info(msg='Transformed has been Saved')
        
        logging.info(msg='>>> Target Feature has been Transformed Successfully <<<')
        
        # Return transformed data <pd.Series>, <pd.Series>
        return y_train_transformed, y_test_transformed


# python src/component/target_transformation.py
if __name__ == '__main__':
    
    logging.info(msg="<<<<< Target Feature Transformation Started >>>>>")
    
    # Fetch Data
    income_data = fetch_data(FILE_NAME='income_data_nona.csv', DIRECTORY_NAME='raw')
    
    # Fit and Transform Data Simple Featured Dataset Using FeatureTransformer Class(For Testing Perpose) 
    # transform_ = FeatureTransformation(X=income_data)
    # transform_.fit(type_='simple_transform', save=True)
    # X_train_transformed, X_test_transformed = transform_.transform(X=income_data, type_='simple_transform')
    
    # Fit and Transform Data Featured Dataset Using FeatureTransformer Class(For Testing Perpose)
    # Extract New Features & Fit Transform Them.
    extractor = FeatureExtractor()
    income_data = extractor.fit_transform(X=income_data)
    
    # Create an Instanse
    transform_ = FeatureTransformation(X=income_data)
    transform_.fit(type_='transform')
    
    # Create an Instanse of Target Transformation and then fit & transform target feature
    target = TargetTransformation(transform_)
    target.fit(save=True)
    y_train_transformed, y_test_transformed = target.transform(income_data['income'])
    
    logging.info(msg=f"y Train Data\n{y_train_transformed.head(4)}")
    logging.info(msg=f"y Test Data\n{y_test_transformed.head(4)}")
    
    logging.info(msg="<<<<< Transformation Successsfully Completed >>>>>")
    