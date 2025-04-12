import os
import joblib
import numpy as np
import pandas as pd
from src.utils import fetch_data
from src.logger import logging
from src.component.feature_transformation import FeatureTransformation
from src.component.hendling_missing_values import TransformData, HandlingMissingValues
from src.component.feature_extraction import FeatureExtractor
from src.component.target_feature_tranformation import TargetFeatureTransform

class ServingPipeline:
    def __init__(self):
        
        logging.info(msg='<<<<Serving Pipeline has been Started>>>>')
        
        encoders = list()
        file_path = os.path.join('models', 'missing_value_imputer')

        for file_name in [
            'column_transformer.pkl',
            'label_encoder_dict.pkl',
            'Random_Forest_Imputuer_Dict.pkl'
            ]:
            full_path = os.path.join(file_path, file_name)
            
            with open(full_path, 'rb') as file:
                encoder = joblib.load(file)
                encoders.append(encoder)

        transformer = list()
        file_path = os.path.join('artifacts', 'column_transformers')
        
        for file_name in [
            'featured_transformer.pkl',
            'simple_featured_transformer.pkl'
            ]:
            full_path = os.path.join(file_path, file_name)
            
            with open(full_path, 'rb') as file:
                t = joblib.load(file)
                transformer.append(t)
        
        models = list()
        file_path = os.path.join('models', 'XGBoost')
        for file_name in [
            os.path.join('XGBoost_featured_set', 'XGBoost_tuned_p0.728868_r0.760399.pkl'),
            os.path.join('XGBoost_simple_featured_set', 'xgboost_tuned_p0.700441_r0.793677.pkl')
            ]:
            full_path = os.path.join(file_path, file_name)
            
            with open(full_path, 'rb') as file:
                model = joblib.load(file)
                models.append(model)
        
        # Transformers and Models to Handling Missing Values
        self.impu_preprocessor = encoders[0]
        self.impu_label_encoder_dict = encoders[1]
        self.impu_model = encoders[2]
        
        # Transformers
        self.feature_transformer = transformer[0]
        self.simple_feature_transformer = transformer[1]
        
        # Models
        self.featured_model = models[0]
        self.simple_model = models[1]
    
        
    def transform(self, data: pd.DataFrame):
        self.data = data.copy()
        X = data.copy()
        
        # Check if the input data is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame - got {type(X)}")

        impu_transform = TransformData()
        impu_model = HandlingMissingValues(model=self.impu_model)
        
        # Impute missing values
        # X_impu, X_no_impu = impu_transform.prepare_data(X)
        
        X.replace(to_replace='?', value=np.nan, inplace=True)
        
        # Spliting income set based on the missing values for imputation
        X_impu = X[X.isna().any(axis=1)].copy()
        X_no_impu = X.dropna().copy()
        
        if not X_impu.empty:
            
            # transform(...) -> X_transform, transformer
            _, impu_preprocessor = impu_transform.transform(X=X_impu, preprocessor=self.impu_preprocessor)
            X_no_impu = impu_model.predict(
                X=X_impu,
                preprocessor=impu_preprocessor,
                label_encoder_dict=self.impu_label_encoder_dict
                )
        
        X_no_impu.columns = [col.replace('-', '_').strip() for col in X_no_impu.columns]
        
        # Obtaining New Features From Given Ones & Apply Transformation
        extractor = FeatureExtractor()
        X_featured = extractor.fit_transform(X=X_no_impu)
        
        feature_transform_ = FeatureTransformation(
            X=X_featured,
            preprocessor=self.feature_transformer,
            training=False
            )
        X_1, _ = feature_transform_.transform(X=X_featured, type_='transform')
        
        featured_target_encode = TargetFeatureTransform(type_='transform', training=False)
        X_1 = featured_target_encode.transform(X=X_1)
        
        # Apply Simple Feature Transformation
        simple_feature_transform_ = FeatureTransformation(
            X=X_no_impu,
            preprocessor=self.simple_feature_transformer,
            training=False
            )
        X_2, _ = simple_feature_transform_.transform(X=X_no_impu, type_='simple_transform')
        
        simple_target_encode = TargetFeatureTransform(type_='simple_transform', training=False)
        X_2 = simple_target_encode.transform(X=X_2)
        
        self.X_1 = X_1
        self.X_2 = X_2
        
        # return self.X_1, self.X_2
    
    def predict(self):
        
        logging.info(msg='Priedicting on Given Sample')
        
        pred_1 = self.featured_model.named_steps['xgboost_classifier'].predict(self.X_1)
        pred_2 = self.simple_model.named_steps['xgboost_classifier'].predict(self.X_2)
        
        logging.info(msg='Prediction is Completed')
        
        
        pred = []
        
        for p1, p2 in zip(pred_1, pred_2):
            if (p1 == 1) ^ (p2 == 1):
                pred.append('>50K')
            else:
                pred.append('<=50K' if p1 == 0 else '>50K')
            
        self.data['pred_income'] = pred
        
        logging.info(msg='<<<<Serving Pipeline has been Completed>>>>')
            
        return self.data
    

# Test Your Module: python src/pipeline/serving_pipeline.py
if __name__ == '__main__':
    
    # Fetch the data
    income_data = fetch_data(FILE_NAME='income_evaluation.csv', DIRECTORY_NAME='raw')
    
    # Choose One or Multiple Rows od Your Choice For the Test Purpose
    rows_to_transform = pd.DataFrame([income_data.dropna().iloc[7]])
    # rows_to_transform = income_data.dropna().iloc[0:10]

    # Initialize the ServingPipeline
    serving_pipeline = ServingPipeline()

    # Transform the selected rows
    # X_1, X_2 = serving_pipeline.transform(rows_to_transform)
    # print(X_1.shape, X_2.shape)

    # Now Transform & Predict on the selected rows
    serving_pipeline.transform(rows_to_transform)
    pred_data = serving_pipeline.predict()
    
    logging.info(f'Imputed Set\n{pred_data}')
    