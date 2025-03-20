import joblib
import os
import json
import pandas as pd
import numpy as np
from src.utils import transforme_DataFrame, fetch_data
from src.logger import logging
from typing import Union, Optional, Literal, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from src.component.transformation import FrequencyEncoder, Winsorizer
from src.component.feature_extraction import FeatureExtractor
from sklearn.model_selection import train_test_split

# This Columns trasnformer support only two types of feature sets,
# where we name it transform and simple transform. type_ is a parameter that accept only these two str values
def is_valied(type_) -> None:
    
    if not isinstance(type_, str):
        raise TypeError(f'type_ must be a string - got: {type(type_)}')
    
    if  type_ not in ['simple_transform', 'transform']:
        raise ValueError(f'type_ must be either simple_transform or transform - got: {type_}')
    
    return None

# Split Data Into Train and Test Sets
def train_test_split_(X, features) -> dict:
    
    if 'income' not in X.columns:
        raise KeyError("Target 'income' columns is missing")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['income']),
        X['income'],
        test_size=0.15,
        random_state=42
    )
                
    X_train_dropped = X_train.drop(columns=features['target_features'])
    X_test_dropped = X_test.drop(columns=features['target_features'])
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_dropped': X_train_dropped,
        'X_test_dropped': X_test_dropped
    }

# This function is used to create a ColumnTransformer object
def fit_transformer(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    features: dict,
    params: dict,
    type_: Literal['simple_transform', 'transform']
    ):
    
    # Initializing 'i' for slicing according to the type_
    if type_ == 'transform':
        i = -1
    else:
        i = 1
    
    try:
        
        preprocessor = ColumnTransformer(
            transformers=[
                
                ('onehot', OneHotEncoder(), features['onehot_features']),
                
                ('ordinal', OrdinalEncoder(
                    categories=list(params.values())[:i]
                    ), features['ordinal_features']),
                
                ('frequency', FrequencyEncoder(), features['frequency_features']),
                
                ('winsorizer', Winsorizer(
                    feature_limits=params['winsorize_limit']
                    ), features['winsorize_features']),
                
                ('minmax', MinMaxScaler(), features['scale_features']),
                
                ('reminders', 'passthrough', features['remander_features'])
                # We will use target encoder duting the training and validation process.
            ]
        )
        
        preprocessor.fit(X=X, y=y)
        
    except (KeyError, ValueError, TypeError, Exception) as e:
        logging.error(f"An error occurred during the fit process: {e}")
        raise # RuntimeError(f"An error occured during the fit process: {e}") from e
    
    return preprocessor

# This function is used to transfrom a ColumnTransformer object.
def transform_data(X, preprocessor: ColumnTransformer):
    try:
        
        return preprocessor.transform(X=X)
    
    except (KeyError, ValueError, TypeError, Exception) as e:
        logging.error(f"An error occurred during the transform process: {e}")
        raise

# def save_trans_data(X, file_name: str, file_path:str):
#     with open(os.path.join(file_path, file_name), 'w') as file:
#         joblib.dump(X, file)

# Function is to save ColumnTransformer object.
def save_trans(trans, file_name, file_path):
    with open(os.path.join(file_path, file_name), 'wb') as file:
        joblib.dump(trans, file)

# Prepare dataset by:
# - Initializing Features Names to the Custom Encoders.
# - Converting them into Pandas DataFrames.
# - Removing Duplicated Features.
# - Concatinating all component into a single Pandas DataFrame
def prepare_data(
    X_train,
    X_test,
    X_train_transformed,
    X_test_transformed,
    preprocessor,
    features
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    logging.info(msg='Data Preparation Started')
    
    # Spet the name for the columns
    for name, cols in zip(
        ['frequency', 'winsorizer'],
        [features['frequency_features'], features['winsorize_features']]
    ):
        preprocessor.named_transformers_[name].set_feature_names(cols)
    
    # Converte them into a DataFrame
    X_train_transformed = transforme_DataFrame(transformed=X_train_transformed, preprocessor=preprocessor)
    X_test_transformed = transforme_DataFrame(transformed=X_test_transformed, preprocessor=preprocessor)
    
    # Drop Duplicated Winsorize Features by Selecting there Scale Version
    train_duplicated_features = X_train_transformed[features['winsorize_features']]
    test_duplicated_features = X_test_transformed[features['winsorize_features']]
    
    scaled_train_features = train_duplicated_features.loc[
        :,
        (train_duplicated_features.ge(0) & train_duplicated_features.le(1)).all()
    ]
    scaled_test_features = test_duplicated_features.loc[
        :,
        (test_duplicated_features.ge(0) & test_duplicated_features.le(1)).all()
    ]
    
    X_train_transformed.drop(columns=features['winsorize_features'], axis=1, inplace=True)
    X_test_transformed.drop(columns=features['winsorize_features'], axis=1, inplace=True)
    
    # Concatenate all brocken sets based on features
    X_train_transformed = pd.concat(
        [
            scaled_train_features,
            X_train[features['target_features']].reset_index(drop=True),
            X_train_transformed
        ],
        axis=1
    )
    X_test_transformed = pd.concat(
        [
            scaled_test_features,
            X_test[features['target_features']].reset_index(drop=True),
            X_test_transformed
        ],
        axis=1
    )
    
    logging.info(msg='Data Preparation Completed')
    
    # Return <Train DataFrame> & <Test DataFrame>
    return X_train_transformed, X_test_transformed    
    

class FeatureTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        
        if not isinstance(X, pd.DataFrame):
            # logging.error()
            raise TypeError(f'X must be a Pandas DataFrame - got: {type(X)}')

        self.preprocessor = None
        
        # Load All Data Configurations Files For Column Transformation
        file_names = {
            'mappings': None,
            'transform_parameters': None,
            'transform_features': None,
            'simple_transform_features': None
            }
        
        for file in file_names.keys():
            with open(f'config/data_config/{file}.json', 'r') as json_file:
                file_names[file] = json.load(json_file)
        
        self.mapping = file_names['mappings']
        self.transform_params = file_names['transform_parameters']
        self.transform_features = file_names['transform_features']
        self.simple_transform_features = file_names['simple_transform_features']
        
        # Split Data Into Train & Test Sets
        splitted_dict = train_test_split_(X=X, features=self.transform_features)
        self.X_train, self.X_test = splitted_dict['X_train'], splitted_dict['X_test']
        self.y_train, self.y_test = splitted_dict['y_train'], splitted_dict['y_test']
        self.X_train_dropped, self.X_test_dropped = splitted_dict['X_train_dropped'], splitted_dict['X_test_dropped']

        
    def fit(self,
            type_: Optional[Literal['simple_transform', 'transfrom']] = None,
            save: bool = False
            ):
        
        logging.info(msg='***** Feature Transformation Started *****')
        
        # Check the validity of string type literal
        is_valied(type_=type_)
    
        trans_file_path = 'artifacts/column_transformers'
        
        logging.info(msg='>>> Fitting ColumnTransformer Started. <<<')
        
        if type_ == 'transform':
                                 
            try:
                
                # Fitting a Transformer on Featured Dataset.     
                self.preprocessor = fit_transformer(
                    X=self.X_train_dropped,
                    y=self.y_train,
                    features=self.transform_features,
                    params=self.transform_params,
                    type_ = type_
                    )
            
            except Exception as e:
                raise(f"An error occurred: {e}")
            
            # Save Transformer if True  
            if save:                
                save_trans(
                    trans=self.preprocessor,
                    file_name='featured_transformer.pkl',
                    file_path=trans_file_path
                )
                
                logging.info(msg='Transformer has been Saved')
                
            
        else:
                        
            try:
                
                # Fitting a Transformer on Simple Featured Dataset.
                self.preprocessor = fit_transformer(
                    X=self.X_train_dropped,
                    y=self.y_train,
                    features=self.simple_transform_features,
                    params=self.transform_params,
                    type_=type_
                )
                
            except Exception as e:
                raise(f"An error occurred: {e}")
            
            
            # Save Transformer if True  
            if save:
                save_trans(
                    trans=self.preprocessor,
                    file_name='simple_featured_transformer.pkl',
                    file_path=trans_file_path
                )
                
                logging.info(msg='Transformer has been Saved')
                
        logging.info(msg='>>> ColumnTransformer has been Fitted. <<<')
        
        # Return <sklearn.compose.ColumnTransformer>
        # return preprocessor
                
    def transform(
        self,
        X: Optional[pd.DataFrame] = None,
        type_: Optional[Literal['simple_transform', 'transfrom']] = None,
        save: bool = False,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        logging.info(msg='>>> Transforming Data Started <<<')
        
        if not isinstance(self.preprocessor, ColumnTransformer):
            raise ValueError("Input must be a ColumnTransformer object")
        
        # Check the Validity of string type literal
        is_valied(type_=type_)
        
        data_file_path = 'data'
            
        try:
            
            # Transforming the Dataset using the Fitted Transformer.
            X_train_transformed = transform_data(X=self.X_train_dropped, preprocessor=self.preprocessor)
            X_test_transformed = transform_data(X=self.X_test_dropped, preprocessor=self.preprocessor)
        
        except Exception as e:
            raise(f"An error occurred: {e}")
        
        if type_ == 'transfrom':
            
            # Check the Docs by Press Ctrl + Left Click
            X_train_transformed, X_test_transformed= prepare_data(
                X_train=self.X_train,
                X_test=self.X_test,
                X_train_transformed=X_train_transformed,
                X_test_transformed=X_test_transformed,
                preprocessor=self.preprocessor,
                features=self.transform_features
                )
            
            # Save Transformed Data if True
            if save:
                X_train_transformed.to_csv(
                    os.path.join(data_file_path, 'featured', 'X_train_transformed.csv'),
                    index=False
                    )
                X_test_transformed.to_csv(
                    os.path.join(data_file_path, 'featured', 'X_test_transformed.csv'),
                    index=False
                    )
                
                logging.info(msg='Transformed Train and Test Datasets Saved')
                
        else:
            
            # Check the Docs by Press Ctrl + Left Click
            X_train_transformed, X_test_transformed = prepare_data(
                X_train=self.X_train,
                X_test=self.X_test,
                X_train_transformed=X_train_transformed,
                X_test_transformed=X_test_transformed,
                preprocessor=self.preprocessor,
                features=self.simple_transform_features
            )
        
            # Save Transformed Data if True
            if save:
                X_train_transformed.to_csv(
                    os.path.join(data_file_path, 'processed', 'X_train_simple_transformed.csv'),
                    index=False
                    )
                X_test_transformed.to_csv(
                    os.path.join(data_file_path, 'processed', 'X_test_simple_transformed.csv'),
                    index=False
                    )
                
                logging.info(msg='Transformed Train and Test Datasets Saved')
                     
        logging.info(msg='>>> Data Transformation Completed <<<')
        # Return <pd.DataFrame> & <pd.DataFrame>
        return X_train_transformed, X_test_transformed



# command: python src/component/feature_transformation.py
if __name__ == '__main__':
    
    logging.info(msg='<<<<< Testing Class: Feature Transformation Started >>>>>')
    
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
    transform_.fit(type_='transform', save=True)
    X_train_transformed, X_test_transformed = transform_.transform(X=income_data, type_='transform')
    
    
    logging.info(f'Training Data:\n{X_train_transformed.head(4)}')
    logging.info(f'Testing Data:\n{X_test_transformed.head(4)}')
    
    logging.info(msg='<<<<< Testing of Class: Feature Transformation Successfully Completed >>>>>')
    