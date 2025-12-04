"""
TDM Utility Functions Module
=============================
Reusable functions for Term Deposit Marketing analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_curve, auc)
from sklearn.cluster import KMeans


def identify_feature_types(df):
    """
    Identify categorical and numerical features in a DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    
    Returns:
    --------
    tuple : (categorical_features_df, numerical_features_df)
        Two DataFrames containing categorical and numerical features respectively
    """
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    #categorical_features = df.select_dtypes(include=['object', 'category'])
    #numerical_features = df.select_dtypes(include=['int64', 'float64', 'Int64'])
    return categorical_features, numerical_features


def find_unknown_values(df, categorical_features):
    """
    Find categorical features containing 'unknown' values.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    categorical_features : list
        List of categorical feature names
    
    Returns:
    --------
    list : Features containing 'unknown' values
    """
    features_with_unknown = []
    for feature in categorical_features:
        unknown_count = (df[feature] == 'unknown').sum()
        if unknown_count > 0:
            features_with_unknown.append(feature)
            print(f"Column '{feature}' has {unknown_count} 'unknown' values.")
    return features_with_unknown


def encode_target_variable(series_or_df, target_column=None, mapping={'yes': 1, 'no': 0}):
    """
    Encode target variable with specified mapping.
    
    Parameters:
    -----------
    series_or_df : Series or DataFrame
        Input Series or DataFrame containing target variable
    target_column : str, optional
        Name of target column (required if DataFrame)
    mapping : dict
        Encoding mapping
    
    Returns:
    --------
    Series : Encoded target variable
    """
    if isinstance(series_or_df, pd.DataFrame):
        if target_column is None:
            raise ValueError("target_column must be specified for DataFrame input")
        return series_or_df[target_column].map(mapping)
        return series_or_df[target_column].value_counts()
    else:
        return series_or_df.map(mapping)


def visualize_all_features(df, categorical_features, numerical_features, style='dark_background'):
    """
    Create visualizations for all features.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    categorical_features : list
        List of categorical features
    numerical_features : list
        List of numerical features
    style : str
        Matplotlib style name (default: 'dark_background')
    """
    plt.style.use(style)
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        df[feature].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        plt.hist(df[feature].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def plot_individual_features(df, features, style='dark_background', save_figures=False):
    """
    Plot individual feature distributions.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    features : list
        List of features to plot
    style : str
        Matplotlib style name
    save_figures : bool
        Whether to save figures to disk
    """
    plt.style.use(style)
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        if df[feature].dtype in ['object', 'category'] or df[feature].nunique() < 20:
            # Categorical or discrete features - use bar plot
            df[feature].value_counts().sort_index().plot(kind='bar')
            plt.xticks(rotation=45, ha='right')
        else:
            # Numerical features - use histogram
            plt.hist(df[feature].dropna(), bins=30, edgecolor='black')
        
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.tight_layout()
        
        if save_figures:
            plt.savefig(f'../../reports/figures/distribution_of_{feature}.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_feature_distribution_by_target(df, features, target_column='y', 
                                        yes_color='steelblue', no_color='coral',
                                        save_figures=False, save_path='../../reports/figures'):
    """
    Plot feature distributions split by target variable with two subplots.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing features and target
    features : list
        List of feature column names to plot
    target_column : str
        Name of the target column (default: 'y')
    yes_color : str
        Color for 'yes' class (default: 'steelblue')
    no_color : str
        Color for 'no' class (default: 'coral')
    save_figures : bool
        Whether to save figures to disk
    save_path : str
        Directory path for saving figures
    """
    for col in features:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get percentage distributions
        data_yes = df[df[target_column] == 'yes'][col].value_counts(normalize=True) * 100
        data_no = df[df[target_column] == 'no'][col].value_counts(normalize=True) * 100
        
        # Sort by index for consistent ordering
        data_yes = data_yes.sort_index()
        data_no = data_no.sort_index()
        
        # Subplot 1: target = 'yes'
        ax1.bar(range(len(data_yes)), data_yes.values, color=yes_color, alpha=0.8, edgecolor='black')
        ax1.set_xlabel(col, fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title(f'{col} Distribution ({target_column} = yes)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(data_yes)))
        ax1.set_xticklabels(data_yes.index, rotation=45, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Subplot 2: target = 'no'
        ax2.bar(range(len(data_no)), data_no.values, color=no_color, alpha=0.8, edgecolor='black')
        ax2.set_xlabel(col, fontsize=12)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title(f'{col} Distribution ({target_column} = no)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(data_no)))
        ax2.set_xticklabels(data_no.index, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Distribution of {col} by Subscription Status', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_figures:
            plt.savefig(f'{save_path}/distribution_of_{col}_by_subscription.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def calculate_class_distribution(series_or_df, target_column=None):
    """
    Calculate class distribution and percentages.
    
    Parameters:
    -----------
    series_or_df : Series or DataFrame
        Input Series or DataFrame containing target variable
    target_column : str, optional
        Name of target column (required if DataFrame)
    
    Returns:
    --------
    dict : Dictionary with counts and percentages
    """
    if isinstance(series_or_df, pd.DataFrame):
        if target_column is None:
            raise ValueError("target_column must be specified for DataFrame input")
        data = series_or_df[target_column]
    else:
        data = series_or_df
    
    total = len(data)
    class_counts = data.value_counts()
    
    distribution = {}
    print(f"Total samples: {total}")
    for cls, count in class_counts.items():
        percentage = (count / total) * 100
        distribution[cls] = {'count': count, 'percentage': percentage}
        print(f"Class {cls}: {count} ({percentage:.2f}%)")
    
    return distribution


def compute_class_weights_balanced(y):
    """
    Compute balanced class weights.
    
    Parameters:
    -----------
    y : array-like
        Target variable
    
    Returns:
    --------
    dict : Class weights
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {i: weights[i] for i in range(len(weights))}
    
    print("Class weights:", class_weights)
    return class_weights


def convert_day_to_week(df, day_column='day', week_column='week'):
    """
    Convert day of month to week number.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    day_column : str
        Name of day column
    week_column : str
        Name of new week column
    
    Returns:
    --------
    DataFrame : DataFrame with week column
    """
    df[week_column] = df[day_column].apply(lambda x: (int(x) - 1) // 7 + 1)
    return df[week_column].value_counts()


def convert_month_to_quarter(df, month_column='month', quarter_column='quarter', 
                            drop_month=False):
    """
    Convert month names to quarter numbers.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    month_column : str
        Name of month column
    quarter_column : str
        Name of new quarter column
    drop_month : bool
        Whether to drop the month column
    
    Returns:
    --------
    DataFrame : DataFrame with quarter column
    """
    # Map month names to numbers
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    df['month_num'] = df[month_column].str.lower().map(month_mapping)
    df[quarter_column] = pd.cut(df['month_num'], bins=[0, 3, 6, 9, 12], 
                                labels=[1, 2, 3, 4])
    df[quarter_column] = df[quarter_column].astype(int)
    
    # Clean up
    df.drop('month_num', axis=1, inplace=True)
    if drop_month:
        df.drop(month_column, axis=1, inplace=True)
    
    return df

def train_model(pipeline, X_train, y_train):
    """
    Train a model using the provided pipeline.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        The machine learning pipeline
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    
    Returns:
    --------
    Pipeline : Trained pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model and print metrics.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name for display
    
    Returns:
    --------
    dict : Dictionary containing predictions and metrics
    """
    # Validate model type early for clearer errors
    if not hasattr(model, "predict"):
        raise TypeError(
            "evaluate_model expected a trained estimator with a 'predict' method. "
            "You passed a value of type '" + type(model).__name__ + "'. "
            "Tip: call like evaluate_model(model_lr, X_test, y_test, model_name='LogReg') "
            "and avoid passing the model name string as the first argument."
        )

    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'predictions': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : array
        Confusion matrix array
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name for display
    
    Returns:
    --------
    float : AUC score
    """
    # Validate inputs early to catch common call mistakes
    if isinstance(model, str):
        raise TypeError(
            "First argument 'model' must be a trained estimator. "
            "It looks like you passed the model name first. "
            "Correct usage: plot_roc_curve(model, X_test, y_test, model_name='Your Model Name')."
        )

    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return roc_auc

# get_clusters function
def get_clusters(k, data):
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(data)
    predictions = model.predict(data)
    data["class"] = model.labels_
    return data