#!/usr/bin/env python
"""
Feature engineering module for Enhanced Approach 4 IPO prediction model.

This module contains functions to create the necessary features for the 
Approach4 IPO prediction model, mimicking the exact features created 
during training in Enhanced_Approach4.py.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def prepare_features_for_approach4(data, debug=False):
    """
    Process input data to create all features needed for the Approach4 IPO prediction model.
    
    This function replicates the feature engineering steps from Enhanced_Approach4.py to ensure
    consistent data transformation between training and prediction.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input data containing company information
    debug : bool, optional
        If True, print debug information during processing
        
    Returns:
    --------
    enhanced_data : pandas DataFrame
        Data with all required features for the Approach4 model
    """
    if debug:
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {data.columns.tolist()}")
        print("Starting feature engineering process...")
    # Create a copy to avoid modifying the original dataframe
    enhanced_data = data.copy()
    
    # 1. Add funding round maturity indicators if VC Round column exists
    if 'VC Round' in enhanced_data.columns:
        # Map VC Round to a numeric maturity level (higher = more mature)
        round_maturity_map = {
            'Angel': 1,
            '1st Round': 2,
            '2nd Round': 3,
            '3rd Round': 4,
            '4th Round': 5,
            '5th Round': 6,
            '6th Round': 7,
            '7th Round': 8,
            '8th Round': 9,
            '9th Round': 10,
            '11th Round': 11
        }
        
        # Create round maturity feature
        enhanced_data['Round_Maturity'] = enhanced_data['VC Round'].map(round_maturity_map).fillna(0)
        
        # Define early and late rounds
        early_rounds = ['Angel', '1st Round', '2nd Round', '3rd Round']
        late_rounds = ['5th Round', '6th Round', '7th Round', '8th Round', '9th Round', '11th Round']
        
        # Create binary indicators
        enhanced_data['Is_Early_Round'] = enhanced_data['VC Round'].isin(early_rounds).astype(int)
        enhanced_data['Is_Late_Round'] = enhanced_data['VC Round'].isin(late_rounds).astype(int)
    else:
        # Add default values if VC Round is not available
        enhanced_data['Round_Maturity'] = 0
        enhanced_data['Is_Early_Round'] = 0
        enhanced_data['Is_Late_Round'] = 0
    
    # 2. Add time-based features if Deal Date exists
    if 'Deal Date' in enhanced_data.columns:
        try:
            # Convert Deal Date to datetime
            enhanced_data['Deal Date'] = pd.to_datetime(enhanced_data['Deal Date'], errors='coerce')
            
            # Calculate company age at time of deal if Year Founded exists
            if 'Year Founded' in enhanced_data.columns:
                # Make sure Year Founded is numeric
                enhanced_data['Year Founded'] = pd.to_numeric(enhanced_data['Year Founded'], errors='coerce')
                
                # Calculate company age in years at time of deal
                enhanced_data['Company_Age_at_Deal'] = enhanced_data['Deal Date'].dt.year - enhanced_data['Year Founded']
                # Replace negative or extreme values with NaN
                enhanced_data.loc[enhanced_data['Company_Age_at_Deal'] < 0, 'Company_Age_at_Deal'] = np.nan
                enhanced_data.loc[enhanced_data['Company_Age_at_Deal'] > 100, 'Company_Age_at_Deal'] = np.nan
            else:
                enhanced_data['Company_Age_at_Deal'] = np.nan
            
            # Create a feature for the year of the deal
            enhanced_data['Deal_Year'] = enhanced_data['Deal Date'].dt.year
            
            # Create quarter feature (Q1, Q2, Q3, Q4)
            enhanced_data['Deal_Quarter'] = enhanced_data['Deal Date'].dt.quarter
            
            # Initialize Days_Since_Last_Funding
            enhanced_data['Days_Since_Last_Funding'] = np.nan
            
            # If multiple companies, calculate for each company
            if 'Companies' in enhanced_data.columns:
                for company in enhanced_data['Companies'].unique():
                    company_data = enhanced_data[enhanced_data['Companies'] == company].copy()
                    if len(company_data) > 1 and pd.notna(company_data['Deal Date']).all():
                        # Sort by date
                        company_data = company_data.sort_values('Deal Date')
                        # Calculate days difference
                        days_diff = company_data['Deal Date'].diff().dt.days
                        # Update the original dataframe
                        enhanced_data.loc[company_data.index, 'Days_Since_Last_Funding'] = days_diff
        except Exception as e:
            print(f"Error processing date features: {e}")
    else:
        # Add default values if Deal Date is not available
        enhanced_data['Deal_Year'] = datetime.now().year
        enhanced_data['Deal_Quarter'] = (datetime.now().month - 1) // 3 + 1
        enhanced_data['Days_Since_Last_Funding'] = np.nan
        enhanced_data['Company_Age_at_Deal'] = np.nan
    
    # 3. Calculate key ratios
    # Valuation to Deal Size ratio
    if all(col in enhanced_data.columns for col in ['Post Valuation', 'Deal Size']):
        enhanced_data['Valuation_to_Deal_Size_Ratio'] = enhanced_data['Post Valuation'] / enhanced_data['Deal Size'].replace(0, np.nan)
        enhanced_data['Valuation_to_Deal_Size_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        enhanced_data['Valuation_to_Deal_Size_Ratio'] = np.nan
    
    # Pre to Post ratio
    if all(col in enhanced_data.columns for col in ['Pre-money Valuation', 'Post Valuation']):
        enhanced_data['Pre_to_Post_Ratio'] = enhanced_data['Pre-money Valuation'] / enhanced_data['Post Valuation'].replace(0, np.nan)
        enhanced_data['Pre_to_Post_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        enhanced_data['Pre_to_Post_Ratio'] = np.nan
    
    # 4. Add growth rate features
    # Initialize growth rate features
    for col in ['Deal Size', 'Pre-money Valuation', 'Post Valuation']:
        growth_col = f"{col}_Growth"
        enhanced_data[growth_col] = np.nan
    
    # Calculate growth rates if we have multiple rounds per company
    if 'Companies' in enhanced_data.columns:
        for company in enhanced_data['Companies'].unique():
            company_data = enhanced_data[enhanced_data['Companies'] == company].copy()
            
            if len(company_data) > 1 and 'Deal Date' in company_data.columns:
                # Sort by date if available
                company_data = company_data.sort_values('Deal Date')
                
                # Calculate growth rates for key metrics
                for col in ['Deal Size', 'Pre-money Valuation', 'Post Valuation']:
                    if col in company_data.columns:
                        growth_col = f"{col}_Growth"
                        # Calculate percentage growth
                        growth_rates = company_data[col].pct_change() * 100
                        # Update original dataframe
                        enhanced_data.loc[company_data.index, growth_col] = growth_rates
    
    # 5. Market conditions proxy through industry averages
    if 'Primary Industry Sector' in enhanced_data.columns:
        # For single company prediction, use static defaults or industry averages if available
        enhanced_data['Industry_Year_Avg_Valuation'] = np.nan
        enhanced_data['Industry_YoY_Growth'] = np.nan
        enhanced_data['Valuation_to_Industry_Avg_Ratio'] = np.nan
        
        # If we have enough data, calculate industry averages
        if len(enhanced_data) > 5 and 'Deal_Year' in enhanced_data.columns and 'Post Valuation' in enhanced_data.columns:
            # Calculate industry average by year
            industry_year_avg = enhanced_data.groupby(['Deal_Year', 'Primary Industry Sector'])['Post Valuation'].transform('mean')
            enhanced_data['Industry_Year_Avg_Valuation'] = industry_year_avg
            
            # Calculate ratio to industry average
            enhanced_data['Valuation_to_Industry_Avg_Ratio'] = enhanced_data['Post Valuation'] / industry_year_avg.replace(0, np.nan)
            enhanced_data['Valuation_to_Industry_Avg_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Calculate industry growth trend if we have multiple years
            if enhanced_data['Deal_Year'].nunique() > 1:
                # Group by industry and year to get average valuations
                industry_year_data = enhanced_data.groupby(['Primary Industry Sector', 'Deal_Year'])['Post Valuation'].mean().reset_index()
                
                # For each industry, calculate YoY growth
                for industry in industry_year_data['Primary Industry Sector'].unique():
                    industry_data = industry_year_data[industry_year_data['Primary Industry Sector'] == industry].sort_values('Deal_Year')
                    
                    if len(industry_data) > 1:
                        industry_data['YoY_Growth'] = industry_data['Post Valuation'].pct_change() * 100
                        
                        # Map this growth rate back to enhanced_data
                        for _, row in industry_data.iterrows():
                            if pd.notna(row['YoY_Growth']):
                                mask = (enhanced_data['Primary Industry Sector'] == industry) & (enhanced_data['Deal_Year'] == row['Deal_Year'])
                                enhanced_data.loc[mask, 'Industry_YoY_Growth'] = row['YoY_Growth']
    
    # 6. Create interaction features
    # Interaction between early round indicator and company age
    if all(col in enhanced_data.columns for col in ['Is_Early_Round', 'Company_Age_at_Deal']):
        enhanced_data['Early_Round_x_Age'] = enhanced_data['Is_Early_Round'] * enhanced_data['Company_Age_at_Deal']
    else:
        enhanced_data['Early_Round_x_Age'] = np.nan
    
    # Interaction between early round indicator and deal size
    if all(col in enhanced_data.columns for col in ['Is_Early_Round', 'Deal Size']):
        enhanced_data['Early_Round_x_Deal_Size'] = enhanced_data['Is_Early_Round'] * enhanced_data['Deal Size']
    else:
        enhanced_data['Early_Round_x_Deal_Size'] = np.nan
    
    # Interaction between early round and industry growth
    if all(col in enhanced_data.columns for col in ['Is_Early_Round', 'Industry_YoY_Growth']):
        enhanced_data['Early_Round_x_Industry_Growth'] = enhanced_data['Is_Early_Round'] * enhanced_data['Industry_YoY_Growth']
    else:
        enhanced_data['Early_Round_x_Industry_Growth'] = np.nan
    
    # 7. For IPO rounds, create features that capture relationship to early rounds
    if 'Deal Type' in enhanced_data.columns:
        enhanced_data['Is_IPO'] = (enhanced_data['Deal Type'] == 'IPO').astype(int)
    else:
        enhanced_data['Is_IPO'] = 0  # Default to not IPO
    
    # Company had early round is best calculated with multiple companies, but for single prediction:
    enhanced_data['Company_Had_Early_Round'] = enhanced_data['Is_Early_Round'].max()
    enhanced_data['IPO_and_Had_Early_Round'] = enhanced_data['Is_IPO'] * enhanced_data['Company_Had_Early_Round']
    
    # 8. Add company sequential round features
    enhanced_data['Round_Sequence'] = 1  # Default to first round
    
    # If we have multiple entries for companies
    if 'Companies' in enhanced_data.columns:
        for company in enhanced_data['Companies'].unique():
            company_data = enhanced_data[enhanced_data['Companies'] == company]
            
            # If we have date information, sort by date
            if 'Deal Date' in company_data.columns and pd.notna(company_data['Deal Date']).any():
                sorted_indices = company_data.sort_values('Deal Date').index
                enhanced_data.loc[sorted_indices, 'Round_Sequence'] = range(1, len(sorted_indices) + 1)
            # Otherwise if we have round maturity
            elif 'Round_Maturity' in company_data.columns:
                sorted_indices = company_data.sort_values('Round_Maturity').index
                enhanced_data.loc[sorted_indices, 'Round_Sequence'] = range(1, len(sorted_indices) + 1)
    
    # 9. Calculate round progression speed (average days between rounds)
    enhanced_data['Avg_Days_Between_Rounds'] = np.nan
    
    if 'Days_Since_Last_Funding' in enhanced_data.columns and 'Companies' in enhanced_data.columns:
        for company in enhanced_data['Companies'].unique():
            company_data = enhanced_data[enhanced_data['Companies'] == company]
            if len(company_data) > 1:
                avg_days = company_data['Days_Since_Last_Funding'].mean()
                enhanced_data.loc[enhanced_data['Companies'] == company, 'Avg_Days_Between_Rounds'] = avg_days
    
    # 10. Add outlier indicators for extreme valuations
    enhanced_data['Valuation_Z_Score'] = np.nan
    enhanced_data['Is_Valuation_Outlier'] = 0
    
    # If we have enough data, calculate z-scores
    if 'Post Valuation' in enhanced_data.columns and len(enhanced_data) > 5:
        # Calculate z-score if we have enough data points
        mean_val = enhanced_data['Post Valuation'].mean()
        std_val = enhanced_data['Post Valuation'].std()
        
        if std_val > 0:
            enhanced_data['Valuation_Z_Score'] = (enhanced_data['Post Valuation'] - mean_val) / std_val
            enhanced_data['Is_Valuation_Outlier'] = (np.abs(enhanced_data['Valuation_Z_Score']) > 2).astype(int)
    
    # Return the enhanced dataframe with all required features
    return enhanced_data
