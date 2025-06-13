#!/usr/bin/env python
"""
Final test for IPO prediction focusing on resolving the isnan error.
"""
import os
import pandas as pd
import numpy as np
from ml_predictor import predict_ipo_valuation
import traceback
import sys

def test_ipo_prediction_with_sample_data():
    """Test IPO prediction with sample data similar to what the app would use."""
    print("Running final IPO prediction test...")
    
    # Create a sample dataframe similar to what the app would produce
    sample_data = pd.DataFrame({
        'Deal Size': [5000000],
        'Primary Industry Sector': ['Software'],
        'Deal Type': ['Early Stage VC'],
        'Pre-money Valuation': [15000000],
        'Post Valuation': [20000000],
        'Revenue': [2000000],
        'VC Round': ['2nd Round'],
        'Year Founded': [2018],
        'Deal Date': ['2023-05-01'],
        'Employees': [45],
        '# Investors': [3]
    })
    
    print("\nSample data:")
    print(sample_data)
    
    try:
        # Run prediction
        predictions, description = predict_ipo_valuation(sample_data)
        
        print("\nPrediction successful!")
        print("\nPredicted IPO Valuation:")
        print(f"${predictions['Predicted_IPO_Valuation'].iloc[0]:,.2f}")
        
        return True
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        # Specific check for 'isnan' error
        if "ufunc 'isnan'" in str(e):
            print("\nAnalyzing 'isnan' error:")
            try:
                # Create a patch for the InteractionFeatureTransformer class
                print("Checking InteractionFeatureTransformer...")
                from ml_predictor import InteractionFeatureTransformer
                
                # Create a test instance
                transformer = InteractionFeatureTransformer()
                
                # Try a simple transformation with controlled data
                test_df = pd.DataFrame({
                    'a': [1.0, 2.0, 3.0],
                    'b': [4.0, 5.0, 6.0]
                })
                
                print("Testing transformer with clean data...")
                result = transformer.fit_transform(test_df)
                print("Transformation successful with clean data.")
                
                # Now try with some NaN values
                test_df_nan = pd.DataFrame({
                    'a': [1.0, np.nan, 3.0],
                    'b': [4.0, 5.0, np.nan]
                })
                
                print("Testing transformer with NaN data...")
                result_nan = transformer.fit_transform(test_df_nan)
                print("Transformation successful with NaN data.")
                
                print("\nInteractionFeatureTransformer seems to work correctly.")
                
            except Exception as internal_e:
                print(f"Internal error during transformer test: {str(internal_e)}")
        
        return False

if __name__ == "__main__":
    success = test_ipo_prediction_with_sample_data()
    if success:
        print("\n✅ IPO prediction test successful!")
        sys.exit(0)
    else:
        print("\n❌ IPO prediction test failed.")
        sys.exit(1)
