#!/usr/bin/env python
"""
Diagnostic tool to test the IPO prediction model with detailed error handling.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
import logging
from ml_predictor import predict_ipo_valuation

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ipo_model_debug.log')
    ]
)
logger = logging.getLogger('ipo_model_diagnostic')

def test_ipo_model_with_error_handling():
    """Test the IPO prediction model with detailed error handling."""
    logger.info("Starting IPO model diagnostic test")
    
    try:
        # Check if model files exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'saved_approach4_model/approach4_valuation_prediction_model.pkl')
        features_path = os.path.join(base_dir, 'saved_approach4_model/approach4_model_features.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        if not os.path.exists(features_path):
            logger.error(f"Features file not found: {features_path}")
            return False
        
        logger.info("Model files found. Loading feature info...")
        
        # Load feature info for reference
        with open(features_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        logger.info(f"Feature info loaded. Contains {len(feature_info.get('numerical_features', []))} numerical features and {len(feature_info.get('categorical_features', []))} categorical features")
        
        # Create test data
        logger.info("Creating test data...")
        test_data = pd.DataFrame({
            'Deal Size': [1000000],
            'Primary Industry Sector': ['Software'],
            'Deal Type': ['Early Stage VC'],
            'Pre-money Valuation': [5000000],
            'Post Valuation': [6000000],
            'Revenue': [500000],
            'VC Round': ['2nd Round'],
            'Year Founded': [2020],
            'Deal Date': ['2024-01-15'],
            'Employees': [25],
            '# Investors': [2]
        })
        
        logger.info("Test data created with shape: {}".format(test_data.shape))
        logger.debug("Test data columns: {}".format(test_data.columns.tolist()))
        
        logger.info("Running prediction function...")
        try:
            predictions, description = predict_ipo_valuation(test_data)
            
            logger.info("Prediction completed successfully")
            logger.info(f"Predicted IPO Valuation: ${predictions['Predicted_IPO_Valuation'].iloc[0]:,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in predict_ipo_valuation: {str(e)}")
            logger.error("Traceback: {}".format(traceback.format_exc()))
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Traceback: {}".format(traceback.format_exc()))
        return False

if __name__ == "__main__":
    success = test_ipo_model_with_error_handling()
    
    if success:
        logger.info("✅ IPO model diagnostic test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ IPO model diagnostic test failed!")
        sys.exit(1)
