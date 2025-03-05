import math
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from src.exception import CustomException
from src.logger import logging
from typing import Union, Optional
from matplotlib import pyplot as plt

    
def plot_categorical_features(
    data,
    columns,
    # target_col: Optional[str] = None,
    fixed_hue: Optional[str] = None,
    orientation: str = 'v',
    plot_type: str = 'countplot',
    x_axis_label: Union[str, list, np.ndarray] = '',
    y_axis_label: Union[str, list, np.ndarray] = 'Count',
    subplot_title: Union[str, list, np.ndarray] = 'Axis Title',
    main_title: str = 'Figure Title',
    **kwargs
):

    # Convert a single string into a list for compatibility 
    if isinstance(x_axis_label, str):
        x_axis_label = [x_axis_label] * len(columns)
    if isinstance(y_axis_label, str):
        y_axis_label = [y_axis_label] * len(columns)
    if isinstance(subplot_title, str):
        subplot_title = [subplot_title] * len(columns)
        
    # Validate orientation input
    if orientation not in ['v', 'h']:
        raise ValueError('Orientation must be either "v" (vertical) or "h" (horizontal)')
    
    # Number of subplots
    num_col = len(columns)
    cols = 3
    rows = math.ceil(num_col / cols)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    
    
    for i, feature in enumerate(columns):
        # Dynamically call the appropriate seaborn plot function
        plot_func = getattr(sns, plot_type, None)
        if plot_func is None:
            raise ValueError(f'Invalid plot type: {plot_type}. Please use a valid plot type from the seaborn library.')
        
        # Pass the required arguments to the plot function & filter kwargs
        valid_kwargs = {k: v for k, v in kwargs.items() if k not in ['x_axis_label', 'y_axis_label', 'subplot_title']}
        
        # Check if the order is there in valid_kwargs
        if plot_type in ['countplot', 'barplot']:
            if 'order' not in valid_kwargs:
                valid_kwargs['order'] = data[feature].value_counts().index
                # print(valid_kwargs['order'])
        
        hue_param = fixed_hue or feature
        
        if orientation == 'v':
            plot_func(
                data=data,
                x=feature,
                # y=target_col,
                hue=hue_param,
                ax=axes[i],
                **valid_kwargs  # Pass only valid kwargs
            )
        
        elif orientation == 'h':
            plot_func(
                data=data,
                y=feature,
                hue=hue_param,
                ax=axes[i],
                **valid_kwargs  # Pass only valid kwargs
            )
        
        # Set title and labels
        axes[i].set_title(subplot_title[i].title(), fontsize=14, fontweight='light')
        axes[i].set_xlabel(x_axis_label[i].title(), fontsize=12, fontweight='ultralight')
        axes[i].set_ylabel(y_axis_label[i].title(), fontsize=12, fontweight='ultralight')
        
        # Rotate x-tick labels if the feature has many unique values
        if pd.api.types.is_object_dtype(data[feature]):
            if len(data[feature].unique()) < 18:
                axes[i].tick_params(axis='x', rotation=65)
            else:
                axes[i].tick_params(axis='x', rotation=90)
    
    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Properly remove unused axes
    
    # Add a global title and adjust layout
    fig.suptitle(main_title.title(), fontsize=28, fontweight='normal')
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.show()

def fetch_data(FILE_NAME: str, DIRECTORY_NAME: str) -> pd.DataFrame:
    
    '''
    Fetching data from a CSV file and returning it as a pandas DataFrame.
    
    
    Parameter:
    
    - FILE_NAME: Name of the CSV file.
    
    - DIRECTORY_NAME: Name of the directory where the CSV file is located.
    '''
    
    # Setting the basepath
    os.chdir('f:\\Data Science\\ML Projects\\ML Project by Engineering Wala Bhaiya\\ML_Pipeline_Project')
    BASE_PATH = os.getcwd()

    # Importing the dataset from the data source
    try:
        RAW_DATA_PATH = os.path.join(BASE_PATH, 'data', DIRECTORY_NAME)
        income_data = pd.read_csv(os.path.join(RAW_DATA_PATH, FILE_NAME))
    except Exception as e:
        error = CustomException(error_message=e, error_detail=sys)
        logging.info(error.error_message)
        raise e

    # Strpping all columns and values from the object data
    if income_data.select_dtypes(include=['object']).shape[1] > 0:
        income_data.columns = income_data.columns.str.strip()
        temp_df = income_data.select_dtypes(include=['object']).apply(lambda x: x.str.strip())
        income_data.drop(temp_df.columns, axis=1, inplace=True)
        income_data = pd.concat([income_data, temp_df], axis=1)
        del(temp_df)
    
    # Returning pandas DataFrame
    return income_data

def transforme_DataFrame(transformed, preprocessor) -> Union[pd.DataFrame, pd.Series]:  
    all_feature_names = []
    if hasattr(preprocessor, 'transformers_'):
        for _, transformer, features in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names = transformer.get_feature_names_out()
                all_feature_names.extend(feature_names)
            else:
                all_feature_names.extend(features)
        return pd.DataFrame(transformed, columns=all_feature_names)
    
    elif hasattr(preprocessor, 'classes_'):
        return pd.Series(transformed)