import pickle
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin

# Define the CategoricalImputer class needed for unpickling the model
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value='missing'):
        self.fill_value = fill_value
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).fillna(self.fill_value)

def predict_valuation(new_data, model_path='saved_model/valuation_prediction_model.pkl', 
                     features_path='saved_model/model_features.pkl'):
    """
    Predict company valuation using the trained model.
    
    Parameters:
    -----------
    new_data : pandas DataFrame
        Data containing features for prediction
    model_path : str
        Path to the saved model file
    features_path : str
        Path to the saved feature information
        
    Returns:
    --------
    numpy.ndarray
        Predicted valuation(s)
    """
    try:
        # Get the base directory for relative paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_path)
        features_path = os.path.join(base_dir, features_path)
        
        # Load the model and feature information
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        # Ensure all required features are present
        required_features = feature_info['numerical_features'] + feature_info['categorical_features']
        missing_features = [f for f in required_features if f not in new_data.columns]
        
        # Create a copy to avoid modifying the original dataframe
        data = new_data.copy()
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with NaN values
            for feature in missing_features:
                data[feature] = np.nan
        
        # Keep only required features
        data = data[required_features]
        
        # Make prediction (log-transformed)
        y_pred_log = model.predict(data)
        
        # Transform back to original scale
        y_pred = np.expm1(y_pred_log)
        
        return y_pred
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def check_pitchbook_data(df):
    """
    Check if the provided dataframe has sufficient pitchbook data to make predictions.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to check
        
    Returns:
    --------
    tuple
        (is_valid: bool, message: str) where is_valid indicates if the data is sufficient
        and message provides details on what's missing if not valid
    """
    try:
        # Load feature info to get required features
        base_dir = os.path.dirname(os.path.abspath(__file__))
        features_path = os.path.join(base_dir, 'saved_model/model_features.pkl')
        
        # Check if features file exists
        if not os.path.exists(features_path):
            return False, "Model features file not found. Please ensure the model is properly set up."
        
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        # Get required features
        required_numerical = feature_info.get('numerical_features', [])
        required_categorical = feature_info.get('categorical_features', [])
        
        # Check if the dataframe has the required columns
        missing_numerical = [f for f in required_numerical if f not in df.columns]
        missing_categorical = [f for f in required_categorical if f not in df.columns]
        
        # Critical features that must be present
        critical_features = ['Deal Size', 'Primary Industry Sector', 'Deal Type']
        missing_critical = [f for f in critical_features if f not in df.columns]
        
        # Calculate how many features are present
        present_numerical = len(required_numerical) - len(missing_numerical)
        present_categorical = len(required_categorical) - len(missing_categorical)
        total_present = present_numerical + present_categorical
        total_required = len(required_numerical) + len(required_categorical)
        
        # Avoid division by zero
        presence_percentage = 0
        if total_required > 0:
            presence_percentage = (total_present / total_required) * 100
        
        # If missing critical features, return False
        if missing_critical:
            return False, f"Missing critical features: {', '.join(missing_critical)}"
        
        # If less than 50% of features are present, return False
        if presence_percentage < 50:
            return False, f"Insufficient data: only {presence_percentage:.1f}% of required features are present"
        
        # Return True if sufficient data is present
        return True, f"Sufficient data present: {presence_percentage:.1f}% of required features"
    
    except Exception as e:
        return False, f"Error checking pitchbook data: {e}"

def get_required_features():
    """
    Return the list of features required by the model.
    
    Returns:
    --------
    dict
        Dictionary containing feature information
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        features_path = os.path.join(base_dir, 'saved_model/model_features.pkl')
        
        # Check if features file exists
        if not os.path.exists(features_path):
            return {'numerical_features': [], 'categorical_features': [], 'target_variable': 'Post Valuation'}
            
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        return feature_info
        
    except Exception as e:
        print(f"Error getting required features: {e}")
        # Return empty lists as fallback
        return {'numerical_features': [], 'categorical_features': [], 'target_variable': 'Post Valuation'}

def process_pitchbook_data(csv_file):
    """
    Process CSV file with pitchbook data, generate predictions for each row.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file with pitchbook data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and predictions
    """
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        
        # Check if data is valid for prediction
        is_valid, message = check_pitchbook_data(df)
        
        if not is_valid:
            print(f"Warning: {message}")
            # Continue anyway but with a warning
        
        # Make predictions
        predictions = predict_valuation(df)
        
        if predictions is None:
            return None
            
        # Add predictions to the original dataframe
        df['AI_Predicted_Valuation'] = predictions
        
        # Calculate error metrics if target variable exists
        feature_info = get_required_features()
        target_variable = feature_info.get('target_variable')
        
        if target_variable in df.columns:
            valid_mask = df[target_variable].notna()
            
            if valid_mask.any():
                # Calculate errors only for rows with valid target values
                df.loc[valid_mask, 'Absolute_Error'] = np.abs(
                    df.loc[valid_mask, target_variable] - df.loc[valid_mask, 'AI_Predicted_Valuation']
                )
                
                df.loc[valid_mask, 'Percentage_Error'] = (
                    df.loc[valid_mask, 'Absolute_Error'] / df.loc[valid_mask, target_variable]
                ) * 100
        
        return df
        
    except Exception as e:
        print(f"Error processing the pitchbook data: {e}")
        return None