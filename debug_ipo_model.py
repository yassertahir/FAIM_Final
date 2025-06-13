#!/usr/bin/env python
"""
Enhanced test script for debugging the IPO valuation prediction model
"""

import os
import pickle
import pandas as pd
import numpy as np
from ml_predictor import predict_ipo_valuation, OutlierClipper, InteractionFeatureTransformer, CategoricalImputer
from approach4_feature_engineering import prepare_features_for_approach4

def inspect_model_features():
    """Inspect the saved model features to understand what's required"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, 'saved_approach4_model/approach4_model_features.pkl')
    model_path = os.path.join(base_dir, 'saved_approach4_model/approach4_valuation_prediction_model.pkl')
    
    print("\n==== Model Features Inspection ====\n")
    
    try:
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
            
        print("Feature information loaded successfully")
        
        # Print feature categories
        for category, features in feature_info.items():
            if isinstance(features, list):
                print(f"\n{category} ({len(features)} features):")
                if len(features) > 20:
                    print(f"First 10: {features[:10]}")
                    print(f"Last 10: {features[-10:]}")
                else:
                    print(features)
            else:
                print(f"\n{category}: {features}")
        
        # Load model to check expected features
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        if hasattr(model, 'feature_names_in_'):
            print("\nModel expects these features:")
            print(model.feature_names_in_)
        elif hasattr(model, 'steps') and len(model.steps) > 0:
            # Check if it's a pipeline with a final estimator that has feature names
            final_step = model.steps[-1][1]
            if hasattr(final_step, 'feature_names_in_'):
                print("\nModel expects these features:")
                print(final_step.feature_names_in_)
        
        return feature_info
    except Exception as e:
        print(f"Error inspecting model features: {e}")
        return None

def create_test_data():
    """Create comprehensive test data for IPO prediction"""
    test_data = pd.DataFrame({
        'Companies': ['TestCo1', 'TestCo2'],
        'Deal Size': [1000000, 2000000],
        'Primary Industry Sector': ['Software', 'Healthcare'],
        'Deal Type': ['Early Stage VC', 'Late Stage VC'],
        'Pre-money Valuation': [5000000, 10000000],
        'Post Valuation': [6000000, 12000000],
        'Revenue': [500000, 1000000],
        'EBITDA': [100000, 200000],
        'Net Income': [50000, 100000],
        'VC Round': ['2nd Round', '4th Round'],
        'Deal Date': ['2024-01-15', '2024-03-20'],
        'Year Founded': [2020, 2018],
        'Employees': [25, 50],
        '# Investors': [2, 4]
    })
    
    return test_data

def test_feature_engineering():
    """Test the feature engineering function separately"""
    print("\n==== Feature Engineering Test ====\n")
    
    test_data = create_test_data()
    print(f"Original test data shape: {test_data.shape}")
    
    try:
        # Process with our feature engineering function
        enhanced_data = prepare_features_for_approach4(test_data)
        print(f"Enhanced data shape: {enhanced_data.shape}")
        
        # Print new features
        new_features = [col for col in enhanced_data.columns if col not in test_data.columns]
        print(f"\nNew features added ({len(new_features)}):")
        print(new_features)
        
        # Print a sample of the data
        print("\nEnhanced data sample:")
        print(enhanced_data.head())
        
        return enhanced_data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return None

def test_prediction():
    """Test the actual prediction function"""
    print("\n==== Prediction Test ====\n")
    
    test_data = create_test_data()
    
    try:
        predictions, description = predict_ipo_valuation(test_data)
        print("Prediction successful!")
        print("\nPredictions:")
        print(predictions[['Companies', 'Predicted_IPO_Valuation']])
        
        print("\nDescription:")
        print(description[:200] + "..." if len(description) > 200 else description)
        
        return predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("===== ENHANCED IPO MODEL TEST =====")
    
    # First inspect what features we need
    feature_info = inspect_model_features()
    
    # Test feature engineering
    enhanced_data = test_feature_engineering()
    
    # Test the full prediction
    predictions = test_prediction()
    
    if predictions is not None:
        print("\n✅ IPO prediction model test completed successfully!")
    else:
        print("\n❌ IPO prediction model test failed!")
