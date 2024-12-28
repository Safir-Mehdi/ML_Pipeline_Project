import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Union

def plot_categorical_features(
    data,
    columns,
    target_col,
    plot_type: str = 'countplot',
    x_axis_label: Union[str, list, np.ndarray] = '',
    y_axis_label: Union[str, list, np.ndarray] = 'Count',
    subplot_title: Union[str, list, np.ndarray] = 'Axis Title',
    main_title: str = 'Figure Title',
    **kwargs
):
    '''
    Plot the categorical features against the target column using the specified plot type.
    
    Parameters:
    - data: pd.DataFrame
        The dataset containing the features and target column.
        
    - columns: list
        List of categorical features to plot.
        
    - target_col: str
        The target column (e.g., 'income').
        
    - plot_type: str
        Type of seaborn plot to use (e.g., 'countplot', 'boxplot', etc).
        
    - x_axis_label: str
        To set the x-axis label.
    
    - y_axis_label: str
        To set the y-axis label.
        
    - subplot_title: str
        To set the axis title.
    
    - main_title: str
        To set the figure title.
        
    - **kwargs: Additional keyword arguments to pass to the seaborn plot function.
    '''
    # Convert a single string into a list for compatibility 
    if isinstance(x_axis_label, str):
        x_axis_label = [x_axis_label] * len(columns)
    if isinstance(y_axis_label, str):
        y_axis_label = [y_axis_label] * len(columns)
    if isinstance(subplot_title, str):
        subplot_title = [subplot_title] * len(columns)
    
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
        
        plot_func(
            data=data,
            x=feature,
            # y=target_col,
            hue=feature,
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