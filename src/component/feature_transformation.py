import joblib
import os
import json
import pandas as pd
from src.utils import fetch_data
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
    X_test: Optional[pd.DataFrame],
    X_train_transformed,
    X_test_transformed: Optional[pd.DataFrame],
    preprocessor,
    features
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    
    logging.info(msg='Data Preparation Started')
    
    # Check if both are None, and '!=' is working there as an 'XOR'
    if (X_test is None) != (X_test_transformed is None):
        raise ValueError(
            "Exactly one of 'X_test' or 'X_test_transformed' is None. Both must be provided or None."
            )
    
    # Fetch the names from preprocessor
    cols = preprocessor.get_feature_names_out()
    
    # Converte them into a DataFrame
    X_train_transformed = pd.DataFrame(data=X_train_transformed, columns=cols)
    
    # Drop Duplicated Winsorize Features by Selecting there Scale Version
    scaled_train_features = X_train_transformed.loc[
        :,
        ['minmax__'+col for col in features['winsorize_features']]
    ]
    
    columns_to_remove = features['winsorize_features']
    pattern = '|'.join(columns_to_remove)
    X_train_transformed = X_train_transformed.loc[:, ~X_train_transformed.columns.str.contains(pattern)]
    
    # Concatenate all brocken sets based on features
    X_train_transformed = pd.concat(
        [
            scaled_train_features,
            X_train[features['target_features']].reset_index(drop=True),
            X_train_transformed
        ],
        axis=1
    )
    
    if X_test is not None:
        
        # Converte them into a DataFrame
        X_test_transformed = pd.DataFrame(data=X_test_transformed, columns=cols)
        
        # Drop Duplicated Winsorize Features by Selecting there Scale Version
        scaled_test_features = X_test_transformed.loc[
        :,
        ['minmax__'+col for col in features['winsorize_features']]
        ]
        
        columns_to_remove = features['winsorize_features']
        pattern = '|'.join(columns_to_remove)
        X_test_transformed = X_test_transformed.loc[:, ~X_test_transformed.columns.str.contains(pattern)]
        
        # Concatenate all brocken sets based on features
        X_test_transformed = pd.concat(
            [
                scaled_test_features,
                X_test[features['target_features']].reset_index(drop=True),
                X_test_transformed
            ],
            axis=1
        )
    
    logging.info(msg='Data Preparation Completed')
    
    # Return <Train DataFrame> & <Test DataFrame or None>
    return X_train_transformed, X_test_transformed    

def log_mode(mode_msg = str):
    logging.info(
        f'''
        **************************************************
        {mode_msg}
        **************************************************
        ''') 

class FeatureTransformation(BaseEstimator, TransformerMixin):
    def __init__(
        self, X,
        preprocessor: Optional[ColumnTransformer] = None,
        training=True
        ):
        
        if not isinstance(X, pd.DataFrame):
            # logging.error()
            raise TypeError(f'X must be a Pandas DataFrame - got: {type(X)}')

        self.preprocessor = preprocessor
        self.training = training
        
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
        
        # If is a Training Mode & Else is a Serving Mode
        if self.training:
            
            mode = 'Data Transformetion is Running at Training Mode'
            log_mode(mode_msg=mode)
            
            # Split Data Into Train & Test Sets
            splitted_dict = train_test_split_(X=X, features=self.transform_features)
            self.X_train, self.X_test = splitted_dict['X_train'], splitted_dict['X_test']
            self.y_train, self.y_test = splitted_dict['y_train'], splitted_dict['y_test']
            self.X_train_dropped, self.X_test_dropped = splitted_dict['X_train_dropped'], splitted_dict['X_test_dropped']
        else:
            
            if not isinstance(preprocessor, ColumnTransformer):
                raise TypeError(
                    f'preprocessor must be a sklearn.compose.ColumnTransformer - got: {type(preprocessor)}'
                    )
            
            mode = 'Data Transformation is Running at Serving Mode'
            log_mode(mode_msg=mode)
            
            # Just Assign to the X_train (X_pred - According to the code) and X_train_dropped
            self.X_train = X
            self.X_train_dropped = X.drop(columns=self.transform_features['target_features'])
        
    def fit(self,
            type_: Optional[Literal['simple_transform', 'transfrom']] = None,
            save: bool = False
            ):
        
        if not self.training:
            raise RuntimeError(
        "Cannot call 'fit()' in serving mode. "
        "Set 'training=True' or initialize the object in training mode."
        )
        
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
            
            if self.training:
                
                # Transforming the Dataset using the Fitted Transformer.
                X_train_transformed = transform_data(X=self.X_train_dropped, preprocessor=self.preprocessor)
                X_test_transformed = transform_data(X=self.X_test_dropped, preprocessor=self.preprocessor)

            else:
                
                X_train_transformed = transform_data(X=self.X_train_dropped, preprocessor=self.preprocessor)
                
        except Exception as e:
            raise(f"An error occurred: {e}")
        
        if type_ == 'transfrom':
            # Check the Docs by Press Ctrl + Left Click on the function 'prepare_data'
            
            if self.training:
                
                X_train_transformed, X_test_transformed= prepare_data(
                    X_train=self.X_train,
                    X_test=self.X_test,
                    X_train_transformed=X_train_transformed,
                    X_test_transformed=X_test_transformed,
                    preprocessor=self.preprocessor,
                    features=self.transform_features
                    )
            
            else:
                X_train_transformed, X_test_transformed = prepare_data(
                    X_train=self.X_train,
                    X_test=None,
                    X_train_transformed=X_train_transformed,
                    X_test_transformed=None,
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
            
            # Check the Docs by Press Ctrl + Left Click on the function prepare_data
            if self.training:
                
                # For Training
                X_train_transformed, X_test_transformed= prepare_data(
                    X_train=self.X_train,
                    X_test=self.X_test,
                    X_train_transformed=X_train_transformed,
                    X_test_transformed=X_test_transformed,
                    preprocessor=self.preprocessor,
                    features=self.transform_features
                    )
            
            else:
                
                # For Serving
                X_train_transformed, X_test_transformed = prepare_data(
                    X_train=self.X_train,
                    X_test=None,
                    X_train_transformed=X_train_transformed,
                    X_test_transformed=None,
                    preprocessor=self.preprocessor,
                    features=self.transform_features
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
    
    # ***************************************************************************************************
    
    # Fit and Transform Data For 'Simple Featured Dataset' Using FeatureTransformer Class(For Testing Perpose)
    
    # ---------------------------------------------------
    
    # 1. For Training
    # transform_ = FeatureTransformation(X=income_data)
    # transform_.fit(type_='simple_transform', save=True)
    # X_train_transformed, X_test_transformed = transform_.transform(X=income_data, type_='simple_transform')
    
    # ---------------------------------------------------
    
    # 2. For Serving
    # Here you write a code.
    # instance = income_data.iloc[0:1]
    
    # file_path = r'artifacts\column_transformers\simple_featured_transformer.pkl'
    # with open(file_path, 'rb') as file:
    #     preprocessor = joblib.load(file)
    
    # serving_trans = FeatureTransformation(X=instance, preprocessor=preprocessor, training=False)
    # X_train_transformed, X_test_transformed = serving_trans.transform(X=instance, type_='simple_transform')
    
    #  ***************************************************************************************************
    
    # Fit and Transform Data Featured Dataset Using FeatureTransformer Class(For Testing Perpose)
    
    # Extract New Features & Fit Transform Them.
    extractor = FeatureExtractor()
    income_data = extractor.fit_transform(X=income_data)
    
    # ---------------------------------------------------
    
    # # 1. For Training
    # transform_ = FeatureTransformation(X=income_data)
    # transform_.fit(type_='transform')
    # X_train_transformed, X_test_transformed = transform_.transform(X=income_data, type_='transform')
    
    # ---------------------------------------------------
    
    # 2. For Serving
    instance = income_data.iloc[0:1]
    
    file_path = r'artifacts\column_transformers\featured_transformer.pkl'
    with open(file_path, 'rb') as file:
        preprocessor = joblib.load(file)
    
    serving_trans = FeatureTransformation(X=instance, preprocessor=preprocessor, training=False)
    X_train_transformed, X_test_transformed = serving_trans.transform(X=instance, type_='transform')
 
    
    logging.info(f'Training Data:\n{X_train_transformed.head(4)}')
    logging.info(f'{X_train_transformed.columns}')
    
    if X_test_transformed is not None:
        logging.info(f'Testing Data:\n{X_test_transformed.head(4)}')
    
    logging.info(msg='<<<<< Testing of Class: Feature Transformation Successfully Completed >>>>>')
    