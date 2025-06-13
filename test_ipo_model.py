#!/usr/bin/env python
"""
Test script for verifying that the IPO prediction model can be loaded and used correctly.
"""
import os
import pickle
import pandas as pd
import numpy as np
from ml_predictor import predict_ipo_valuation, OutlierClipper, InteractionFeatureTransformer, CategoricalImputer

def test_ipo_model():
    """
    Test the IPO prediction model loading and basic functionality.
    """
    print("Testing IPO prediction model loading...")
    
    # Check if model files exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'saved_approach4_model/approach4_valuation_prediction_model.pkl')
    features_path = os.path.join(base_dir, 'saved_approach4_model/approach4_model_features.pkl')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found at {features_path}")
        return False
    
    print("Model files found. Attempting to load...")
    
    try:
        # Load the model and feature information
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        print("✓ Feature info loaded successfully")
        
        print(f"\nModel type: {type(model)}")
        print(f"Numerical features: {feature_info.get('numerical_features', [])[:5]}... ({len(feature_info.get('numerical_features', []))} total)")
        print(f"Categorical features: {feature_info.get('categorical_features', [])[:5]}... ({len(feature_info.get('categorical_features', []))} total)")
        
        # Create a simple test dataframe
        print("\nCreating test data...")
        test_data = pd.DataFrame({
            'Deal Size': [1000000],
            'Primary Industry Sector': ['Software'],
            'Deal Type': ['Early Stage VC'],
            'Pre-money Valuation': [5000000],
            'Revenue': [500000],
            'VC Round': ['2nd Round']
        })
        
        print("\nTest data:")
        print(test_data)
        
        # Test prediction function
        print("\nTesting prediction function...")
        predictions, description = predict_ipo_valuation(test_data)
        print("\nPredictions:")
        print(predictions)
        print("\nDescription:")
        print(description[:200] + "..." if len(description) > 200 else description)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load or use model: {e}")
        return False

if __name__ == "__main__":
    success = test_ipo_model()
    if success:
        print("\n✅ IPO prediction model test completed successfully!")
    else:
        print("\n❌ IPO prediction model test failed!")
