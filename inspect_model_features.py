#!/usr/bin/env python
"""
Script to inspect the saved model features
"""
import pickle
import os

def inspect_model_files():
    """Inspect the saved model files to understand what features are expected"""
    
    # Define paths to model files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(base_dir, 'saved_approach4_model/approach4_model_features.pkl'),
        os.path.join(base_dir, 'saved_approach4_model/approach4_valuation_prediction_model.pkl'),
        os.path.join(base_dir, 'saved_model/model_features.pkl')
    ]
    
    # Check each model file
    for path in model_paths:
        if os.path.exists(path):
            print(f"\nInspecting: {path}")
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    print("Contents (dictionary):")
                    for key, value in data.items():
                        if isinstance(value, list):
                            print(f"  {key}: {len(value)} items")
                            if len(value) > 0:
                                print(f"    - First few: {value[:5]}")
                                print(f"    - Last few: {value[-5:] if len(value) >= 5 else value}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"Contents (type: {type(data)}):")
                    # For scikit-learn models, print feature names if available
                    if hasattr(data, 'feature_names_in_'):
                        print(f"  Feature names: {data.feature_names_in_}")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"\nFile not found: {path}")

if __name__ == "__main__":
    inspect_model_files()
