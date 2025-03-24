import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from src.component.transformation import FrequencyEncoder, Winsorizer
from src.utils import fetch_data, transforme_DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging

def create_pipeline(imputer_strategy, transformer):
    return Pipeline([
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('encoder', transformer)
    ])

# Function is to save ColumnTransformer object.
def save_trans(trans, file_name, file_path):
    with open(os.path.join(file_path, file_name), 'wb') as file:
        joblib.dump(trans, file)
        logging.info(f'Obj has been saved successfully.\nFile Path: {file}')

class TransformData(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        FILE_NAME: str ='income_evaluation.csv',
        DIRECTORY_NAME: str ='raw'
        ):
        # Fetching Data
        self.data = fetch_data(FILE_NAME, DIRECTORY_NAME)
        
        # Way to endoce each column
        self.onehot_features = ['sex', 'race']
        self.ordinal_features = ['education']
        self.frequency_features = ['workclass', 'occupation', 'native-country']
        self.target_features = ['relationship', 'marital-status']
        self.winsorize_featues = ['hours-per-week', 'capital-gain', 'capital-loss']
        limits = [(0.001, 0.988), (0.05, 0.993), (0.00, 0.993)]
        self.feature_limits = {feature: limit for feature, limit in zip(self.winsorize_featues, limits)}
        self.remaining_features = self.data.columns[:3].tolist()
        self.ordinal_y = ['income']
    
    def prepare_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        # Formatting columns properly & Replacing '?' with np.nan
        # self.data.columns = [col.replace('-', '_'). strip() for col in self.data.columns]
        
        # if self.data is None and df is None:
        #     raise ValueError('Data is not found. Please provide the data.')
        
        # print(self.data.head(3)) # For Debugging
        
        
        if df is not None:
            # Create Imputation Set if there is an Missing Values in the Instance
            self.data = df.copy()
            self.data.replace(to_replace='?', value= np.nan, inplace=True)
            imputation_set = self.data[self.data.isna().any(axis=1)].copy()
            return imputation_set, None
        elif self.data is None:
            raise ValueError("No data provided")
        
        # self.data = data.copy()
        self.data.replace(to_replace='?', value=np.nan, inplace=True)
        
        # Spliting income set based on the missing values for imputation
        imputation_set = self.data[self.data.isna().any(axis=1)].copy()
        income_data_nona = self.data.dropna().copy()
        
        # Return <Set That Have To Impute> and <Set That Don't contain NaN Values>
        return imputation_set, income_data_nona
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame],
        save: bool = False
        ) -> Union[pd.DataFrame, ColumnTransformer, dict]:
        
        # Checking the validation of the dataframe 
        if isinstance(X, (pd.DataFrame)):
            logging.info('****Data Transformation Started****')
        elif not isinstance(X, (pd.DataFrame)):
            raise ValueError('Input data must be a pandas DataFrame')
        
        # Extracting y for the target encoding
        y = X['income']
        
        # Define sklearn Column Transformer for different types of encoding
        # Note -- This transformer requires all columns of the dataset to be processed with.
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'onehot',
                        create_pipeline(imputer_strategy='most_frequent', transformer=OneHotEncoder(drop='first',)),
                        self.onehot_features
                        ),
                    
                    (
                        'ordinal',
                        create_pipeline(imputer_strategy='most_frequent', transformer=OrdinalEncoder()),
                        self.ordinal_features
                        ),
                    
                    (
                        'frequency',
                        FrequencyEncoder(fillna=-1),
                        self.frequency_features
                        ),
                    
                    (
                        'target',
                        create_pipeline(imputer_strategy='most_frequent', transformer=TargetEncoder()), 
                        self.target_features
                        ),
                    
                    (
                        'winsorizer',
                        create_pipeline(imputer_strategy='mean', transformer=Winsorizer(feature_limits=self.feature_limits)),
                        # Winsorizer(feature_limits=feature_limits),
                        self.winsorize_featues
                        ),
                    
                    (
                        'remainders',
                        create_pipeline(imputer_strategy='mean', transformer='passthrough'),
                        self.remaining_features
                        ),
                    
                    (
                        'ordinal_y',
                        create_pipeline(imputer_strategy='most_frequent', transformer=OrdinalEncoder()),
                        self.ordinal_y
                        )
                ],
                # remainder='passthrough' # Keep other columns as-is (numetical features)
            )
        except ValueError as e:
            raise ValueError(f'Incomplete data. Please check the features and try again.\nError: {e}')
        
        preprocessor.fit(X, y)
        X_transformed = preprocessor.transform(X)
        
        preprocessor.named_transformers_['frequency'].set_feature_names(self.frequency_features)
        preprocessor.named_transformers_['winsorizer'].named_steps['encoder'].set_feature_names(self.winsorize_featues)
        
        # Let's encode target features that have to be impute by model
        workclass_label_encoder = LabelEncoder()
        occupation_label_encoder = LabelEncoder()
        native_country_label_encoder = LabelEncoder()

        label_encoder_dict = {
            'workclass': workclass_label_encoder,
            'occupation': occupation_label_encoder,
            'native-country': native_country_label_encoder
            }

        for col, encode in label_encoder_dict.items():
            encode.fit(X[col])
                
        if save:
            
            file_path = os.path.join('models', 'missing_value_imputer')
            
            # Save the preprocessor
            save_trans(trans=preprocessor, file_name='column_transformer.pkl', file_path=file_path)
            
            # Save the Label Encoder Dict
            save_trans(trans=label_encoder_dict, file_name='label_encoder_dict.pkl', file_path=file_path)
        
        X_transformed = transforme_DataFrame(transformed=X_transformed, preprocessor=preprocessor)  
        logging.info('****Data Transformation Completed****')
        
        # Return <Transformed Data>, <Column Transformer>, <Label Encoder Dict>
        return (X_transformed, preprocessor, label_encoder_dict)
    
    def transform(
        self,
        X: Union[pd.DataFrame],
        preprocessor: ColumnTransformer
        ) -> Union[pd.DataFrame, ColumnTransformer]:
        
        # Checking the validation of the dataframe 
        if isinstance(X, (pd.DataFrame)):
            logging.info('****Data Transformation Started****')
        elif not isinstance(X, (pd.DataFrame)):
            raise ValueError('Input data must be a pandas DataFrame')
        
        # Transform the data using preprocessor
        X_transformed = preprocessor.transform(X)
        
        # Set the names of the features for the frequency and winsorizer transformers
        preprocessor.named_transformers_['frequency'].set_feature_names(self.frequency_features)
        preprocessor.named_transformers_['winsorizer'].named_steps['encoder'].set_feature_names(self.winsorize_featues)
        
        logging.info('****Data Transformation Completed****')
        
        # Return <Transformed Data>, <Column Transformer>
        return X_transformed, preprocessor
        

class HandlingMissingValues:
    def __init__(self, model: dict = None):
        
        # Note -- This class required model to handle missing values
        if model is None:
            raise ValueError('Please provide a model to impute missing values.')
        
        self.model = model
    
    def predict(
        self,
        X: pd.DataFrame = None,
        preprocessor: ColumnTransformer = None,
        label_encoder_dict: dict = None
        ) -> pd.DataFrame:
        
        required_columns = ['occupation', 'workclass', 'native-country']
                
        # Check if parameters are not gonna be None
        # This following condition is same as this condition: X is None or preprocessor is None or label_encoder_dict is None
        if any(params is None for params in [X, preprocessor, label_encoder_dict]):
            raise ValueError('You must provide all the required parameters')
        
        # Check if the required features are in the dataset, if no then raise ValueError
        for col in required_columns:
            if col not in self.model or not isinstance(self.model[col], RandomForestClassifier):
                raise ValueError(f'Model for {col} is not found, & It the model most be a RandomForestClassifier.')
        
        # Check if there is any missing values in the dataset
        if not any(X.isna().any(axis=1)):
            logging.info('****No missing values found in the dataset.****')
            return X
        
        logging.info('****Prediction Started****')
        
        # Predict the missing values by record
        for row in range(len(X)):
            # Get the record
            record = X.iloc[row]
            record = record.to_frame().T
            # Transfrom it
            i_record = transforme_DataFrame(
                transformed=preprocessor.transform(record),
                preprocessor=preprocessor
                )
            # Go through each required column and predict the missing values
            for col in required_columns:
                if (i_record[col] == -1).any(): # -1 indicates missing value
                    try:
                        i_feature_set = i_record.drop(columns=col)  # Drop target column
                        y_pred = self.model[col].predict(i_feature_set) # Predict the missing value
                        X[col].iloc[row] = label_encoder_dict[col].inverse_transform(y_pred)[0] # Assign to the orignal dataframe
                    except NotFittedError as e:
                        raise NotFittedError(f'Model for {col} is not fitted. Please fit the model first before predicting.\nError: {e}')
                    except ValueError as e:
                        raise ValueError(f'Error while predicting {col}.Please check the input data and model\nError: {e}')
                    except Exception as e:
                        raise Exception(f'An unexpected error occurred while predicting {col}.\nError: {e}')
            
        logging.info('****Prediction Completed****')
        
        # Return <DataFrame with Imputed Missing Values>
        return X
    
# From Testing the code: src/component/hendling_missing_values.py
if __name__ == '__main__':
    
    logging.info('****Testing Started****')
    
    # Create an object of TransformData
    transform = TransformData()
    
    # Load model & create an object 
    with open('models/missing_value_imputer/Random_Forest_Imputuer_Dict.pkl', 'rb') as file:
        model_dict = joblib.load(file)
    
    imputer = HandlingMissingValues(model=model_dict)
    
    
    # If you have to provide the data through class (For Development)
    # imputation_set, income_data_nona = transform.prepare_data()
    # _, preprocessor, label_encoder_dict = transform.fit_transform(X=income_data_nona, save=True)
    
    
    # If you have to provide data directly through method of the class (For Serving)
    income_data = fetch_data(FILE_NAME='income_evaluation.csv', DIRECTORY_NAME='raw')
    
    encoders = list()
    filepath = 'models/missing_value_imputer/'
    for file_name in ['column_transformer.pkl', 'label_encoder_dict.pkl']:
        with open(filepath + file_name, 'rb') as file:
            encoders.append(joblib.load(file))
    
    preprocessor = encoders[0]
    label_encoder_dict = encoders[1]
    
    imputation_set, _ = transform.prepare_data(df=income_data)
    _, preprocessor = transform.transform(X=imputation_set, preprocessor=preprocessor)
    
    # Predict the missing values
    logging.info(f'Passed <{len(imputation_set.head(10))}> Rows for Prediction')
    X = imputer.predict(X=imputation_set.head(10), preprocessor=preprocessor, label_encoder_dict=label_encoder_dict)
    logging.info(f'Shape:{X.shape}')
    logging.info(f'{X.head(10)}')
    
    
    # What if there no missing values in the dataset (For Testing)
    # X = imputer.predict(X=imputation_set.head(10), preprocessor=preprocessor, label_encoder_dict=label_encoder_dict)
    
    logging.info('****Testing Completed****')