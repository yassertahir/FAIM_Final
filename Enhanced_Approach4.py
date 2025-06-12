"""
Enhanced Approach 4 for IPO Valuation Prediction

This implementation builds on Enhanced Approach 3 but makes a key difference:
- Uses ALL non-IPO funding rounds for training (not just the latest round)
- Tests on ALL IPO entries
- Incorporates all the enhanced features from Approach 3
- Adds additional outlier handling techniques
- Uses robust scaling and metrics for better performance with extreme values

This approach is designed to leverage maximum data for learning valuation patterns
across all funding stages, which should help better predict the large valuation jumps
that occur between early funding rounds and IPO valuations.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Custom transformer for handling missing values in categorical features
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value='missing'):
        self.fill_value = fill_value
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).fillna(self.fill_value)

# Definition for InteractionFeatureTransformer
class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Generate interaction terms between numerical features"""
    def __init__(self, interaction_only=True, include_bias=False):
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(
            degree=2,  # Just pairwise interactions
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.feature_names_out = None
        
    def fit(self, X, y=None):
        # Convert to numpy array if it's a DataFrame
        X_array = X.values if hasattr(X, 'values') else X
        
        # Fit polynomial features
        self.poly.fit(X_array)
        
        # Store feature names for later use
        if hasattr(X, 'columns'):
            # If X is a DataFrame, get column names
            original_features = X.columns.tolist()
        else:
            # If X is a numpy array, generate feature names
            original_features = [f'x{i}' for i in range(X_array.shape[1])]
            
        self.feature_names_out = self.poly.get_feature_names_out(original_features)
        
        return self
    
    def transform(self, X):
        # Convert to numpy array if it's a DataFrame
        X_array = X.values if hasattr(X, 'values') else X
        
        # Transform data to include interaction terms
        X_poly = self.poly.transform(X_array)
        
        # Return as DataFrame with proper column names
        if hasattr(X, 'index'):
            return pd.DataFrame(X_poly, columns=self.feature_names_out, index=X.index)
        else:
            return pd.DataFrame(X_poly, columns=self.feature_names_out)
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out

# Custom outlier handling transformer
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        
    def fit(self, X, y=None):
        # Get bounds for each feature
        if hasattr(X, 'quantile'):  # DataFrame
            self.lower_bounds_ = X.quantile(self.lower_quantile)
            self.upper_bounds_ = X.quantile(self.upper_quantile)
        else:  # NumPy array
            X_df = pd.DataFrame(X)
            self.lower_bounds_ = X_df.quantile(self.lower_quantile)
            self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self
    
    def transform(self, X):
        # Convert to DataFrame if needed
        if not hasattr(X, 'clip'):
            X = pd.DataFrame(X)
            
        # Clip values
        X_clipped = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_clipped

# Function to calculate Mean Absolute Percentage Error (MAPE) with improved handling of edge cases
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    
    # Filter out invalid values
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred) & mask
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    if len(y_true_valid) == 0:
        return float('nan')  # Return NaN if no valid data points
    
    # Calculate MAPE on valid data
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
    return mape

# Function to calculate Median Absolute Percentage Error (MdAPE) - more robust to outliers than MAPE
def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    
    # Filter out invalid values
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred) & mask
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    if len(y_true_valid) == 0:
        return float('nan')  # Return NaN if no valid data points
    
    # Calculate MdAPE on valid data
    mdape = np.median(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
    return mdape

def enhance_approach4():
    """
    Enhanced implementation of Approach 4 which uses ALL non-IPO entries for training
    and tests on ALL IPO entries.
    """
    print("Enhanced Approach 4: Using ALL non-IPO entries for training")    # Load the dataset
    print("\nLoading and preparing dataset...")
    try:
        df = pd.read_csv('combined_ipo_with_urls.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Number of unique companies: {df['Companies'].nunique()}")
        
        # Display information about the VC Round column
        if 'VC Round' in df.columns:
            print("\nVC Round column analysis:")
            print("Unique values:", df['VC Round'].unique())
            print("\nValue counts:")
            print(df['VC Round'].value_counts(dropna=False))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Define the target variable
    target_variable = 'Post Valuation'
    print(f"Target variable: {target_variable}")

    # 1. Data Preparation
    print("\n1. Preparing data for machine learning...")

    # Identify features for modeling based on domain knowledge
    financial_features = [
        'Deal Size', 'Pre-money Valuation',
        'Revenue', 'EBITDA', 'Net Income', 'Gross Profit',
        'Employees', '# Investors', '# New Investors', '# Follow-on Investors',
        'Raised to Date', 'Total Invested Equity'
    ]

    company_features = [
        'Primary Industry Sector', 'Primary Industry Group', 
        'Current Business Status', 'Current Financing Status',
        'Year Founded'
    ]

    deal_features = [
        'Deal Type', 'VC Round', 'Financing Status'
    ]

    # Combine all potential features
    potential_features = financial_features + company_features + deal_features

    # Filter to only include columns that actually exist in our dataset
    selected_features = [col for col in potential_features if col in df.columns]
    print(f"Selected {len(selected_features)} features for modeling:")
    print(selected_features)

    # Remove rows where target variable is missing
    valid_data = df.dropna(subset=[target_variable]).reset_index(drop=True)
    print(f"\nRows with valid target: {len(valid_data)} (out of {len(df)} total rows)")

    # 2. Feature Engineering - Add new features
    print("\n2. Feature Engineering - Adding new features to handle valuation jumps...")

    # Make a copy of valid_data to work with
    enhanced_data = valid_data.copy()

    # 2.1 Add funding round maturity indicators
    print("- Adding funding round maturity indicators...")
    
    # Map VC Round to a numeric maturity level (higher = more mature)
    # Updated to match the actual values in the dataset
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

    # Helper function to flexibly match VC round values
    def match_vc_rounds(series, patterns, case_sensitive=False):
        """
        Match VC round values against patterns, with option for case sensitivity.
        Returns a boolean Series indicating matches.
        """
        if not case_sensitive:
            # Convert all strings to lowercase for case-insensitive matching
            series = series.str.lower() if hasattr(series, 'str') else series
            patterns = [p.lower() if isinstance(p, str) else p for p in patterns]
        
        # First try exact matches
        matches = series.isin(patterns)
        
        # If we found some matches, return
        if matches.sum() > 0:
            return matches
        
        # If no exact matches, try partial string matching
        if hasattr(series, 'str'):
            for pattern in patterns:
                if isinstance(pattern, str):
                    # Find partial matches
                    partial_matches = series.str.contains(pattern, na=False)
                    matches = matches | partial_matches
        
        return matches
    
    # Create a new feature for funding round maturity
    if 'VC Round' in enhanced_data.columns:
        enhanced_data['Round_Maturity'] = enhanced_data['VC Round'].map(round_maturity_map).fillna(0)
        
        # Define early and late rounds using more robust matching
        early_rounds = ['Angel', '1st Round', '2nd Round', '3rd Round']
        late_rounds = ['5th Round', '6th Round', '7th Round', '8th Round', '9th Round', '11th Round']
        
        # Log what we're doing
        print("\n- Identifying early rounds:", early_rounds)
        print("- Identifying late rounds:", late_rounds)
        
        # Create binary indicators
        enhanced_data['Is_Early_Round'] = match_vc_rounds(enhanced_data['VC Round'], early_rounds).astype(int)
        enhanced_data['Is_Late_Round'] = match_vc_rounds(enhanced_data['VC Round'], late_rounds).astype(int)
        
        # Print how many rows were tagged
        print(f"- Tagged {enhanced_data['Is_Early_Round'].sum()} rows as early rounds")
        print(f"- Tagged {enhanced_data['Is_Late_Round'].sum()} rows as late rounds")
    else:
        print("Warning: 'VC Round' column not found. Skipping round maturity features.")
        
        # Add fallback default values to avoid downstream errors
        enhanced_data['Round_Maturity'] = 0
        enhanced_data['Is_Early_Round'] = 0
        enhanced_data['Is_Late_Round'] = 0

    # 2.2 Add time-based features
    print("- Adding time-based features...")
    
    if 'Deal Date' in enhanced_data.columns:
        # Convert Deal Date to datetime
        try:
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
            
            # Calculate days since last funding round for each company
            # Group by company and sort by date
            enhanced_data['Days_Since_Last_Funding'] = np.nan
            
            for company in enhanced_data['Companies'].unique():
                company_data = enhanced_data[enhanced_data['Companies'] == company].copy()
                if len(company_data) > 1 and pd.notna(company_data['Deal Date']).all():
                    # Sort by date
                    company_data = company_data.sort_values('Deal Date')
                    # Calculate days difference
                    days_diff = company_data['Deal Date'].diff().dt.days
                    # Update the original dataframe
                    enhanced_data.loc[company_data.index, 'Days_Since_Last_Funding'] = days_diff
            
            # Create a feature for the year of the deal
            enhanced_data['Deal_Year'] = enhanced_data['Deal Date'].dt.year
            
            # Create quarter feature (Q1, Q2, Q3, Q4)
            enhanced_data['Deal_Quarter'] = enhanced_data['Deal Date'].dt.quarter
            
        except Exception as e:
            print(f"Error processing date features: {e}")
    else:
        print("Warning: 'Deal Date' column not found. Skipping time-based features.")

    # 2.3 Add growth rate features between funding rounds
    print("- Adding growth rate features between funding rounds...")
    
    # Calculate growth metrics for companies with multiple rounds
    for metric in ['Deal Size', 'Pre-money Valuation', 'Post Valuation']:
        if metric in enhanced_data.columns:
            growth_col_name = f'{metric}_Growth'
            enhanced_data[growth_col_name] = np.nan
            
            for company in enhanced_data['Companies'].unique():
                company_data = enhanced_data[enhanced_data['Companies'] == company].copy()
                
                if len(company_data) > 1:
                    # Sort by round maturity if available, otherwise by date
                    if 'Round_Maturity' in company_data.columns:
                        company_data = company_data.sort_values('Round_Maturity')
                    elif 'Deal Date' in company_data.columns and pd.notna(company_data['Deal Date']).any():
                        company_data = company_data.sort_values('Deal Date')
                    
                    # Calculate percentage growth
                    company_data[growth_col_name] = company_data[metric].pct_change() * 100
                    
                    # Update the original dataframe
                    enhanced_data.loc[company_data.index, growth_col_name] = company_data[growth_col_name]

    # 2.4 Add market condition features (proxy using overall industry trends)
    print("- Adding market condition features...")
    
    # Use the year and industry to capture market trends
    if 'Deal_Year' in enhanced_data.columns and 'Primary Industry Sector' in enhanced_data.columns:
        # Calculate average valuation by year and industry
        year_industry_avg = enhanced_data.groupby(['Deal_Year', 'Primary Industry Sector'])[target_variable].transform('mean')
        enhanced_data['Industry_Year_Avg_Valuation'] = year_industry_avg
        
        # Calculate the ratio of company valuation to industry average that year
        enhanced_data['Valuation_to_Industry_Avg_Ratio'] = enhanced_data[target_variable] / year_industry_avg
        
        # Create a feature for industry growth trend
        industry_year_growth = {}
        
        # Calculate year-over-year growth by industry
        for industry in enhanced_data['Primary Industry Sector'].unique():
            if pd.notna(industry):
                industry_data = enhanced_data[enhanced_data['Primary Industry Sector'] == industry]
                
                if 'Deal_Year' in industry_data.columns:
                    yearly_avg = industry_data.groupby('Deal_Year')[target_variable].mean()
                    yearly_avg = yearly_avg.sort_index()
                    
                    if len(yearly_avg) > 1:
                        # Calculate year over year growth
                        yoy_growth = yearly_avg.pct_change() * 100
                        
                        # Store in dictionary
                        for year, growth in zip(yearly_avg.index[1:], yoy_growth.values[1:]):
                            industry_year_growth[(industry, year)] = growth
        
        # Create a new column for industry YoY growth
        enhanced_data['Industry_YoY_Growth'] = np.nan
        
        # Fill in the values from our dictionary
        for idx, row in enhanced_data.iterrows():
            if pd.notna(row['Primary Industry Sector']) and pd.notna(row['Deal_Year']):
                key = (row['Primary Industry Sector'], row['Deal_Year'])
                if key in industry_year_growth:
                    enhanced_data.loc[idx, 'Industry_YoY_Growth'] = industry_year_growth[key]

    # 2.5 Create interaction features specifically for early rounds
    print("- Creating interaction features for early rounds...")
    
    if 'Is_Early_Round' in enhanced_data.columns:
        # Interaction between early round indicator and company age
        if 'Company_Age_at_Deal' in enhanced_data.columns:
            enhanced_data['Early_Round_x_Age'] = enhanced_data['Is_Early_Round'] * enhanced_data['Company_Age_at_Deal']
        
        # Interaction between early round indicator and deal size
        if 'Deal Size' in enhanced_data.columns:
            enhanced_data['Early_Round_x_Deal_Size'] = enhanced_data['Is_Early_Round'] * enhanced_data['Deal Size']
        
        # Interaction between early round and industry growth
        if 'Industry_YoY_Growth' in enhanced_data.columns:
            enhanced_data['Early_Round_x_Industry_Growth'] = enhanced_data['Is_Early_Round'] * enhanced_data['Industry_YoY_Growth']
            
        # For IPO rounds, create features that capture relationship to early rounds
        if 'Deal Type' in enhanced_data.columns:
            # Create an IPO indicator
            enhanced_data['Is_IPO'] = (enhanced_data['Deal Type'] == 'IPO').astype(int)
            
            # For each company, calculate if they had an early round in the dataset
            company_had_early_round = {}
            for company in enhanced_data['Companies'].unique():
                company_data = enhanced_data[enhanced_data['Companies'] == company]
                had_early = (company_data['Is_Early_Round'] == 1).any()
                company_had_early_round[company] = 1 if had_early else 0
            
            # Add indicator if this company had an early round in the dataset
            enhanced_data['Company_Had_Early_Round'] = enhanced_data['Companies'].map(company_had_early_round)
            
            # Create IPO specific interaction
            enhanced_data['IPO_and_Had_Early_Round'] = enhanced_data['Is_IPO'] * enhanced_data['Company_Had_Early_Round']

    # 2.6 Add company sequential round features
    print("- Adding company sequential round features...")
    
    # Add round sequence number for each company
    for company in enhanced_data['Companies'].unique():
        company_data = enhanced_data[enhanced_data['Companies'] == company]
        
        # If we have date information, sort by date
        if 'Deal Date' in company_data.columns and pd.notna(company_data['Deal Date']).any():
            sorted_indices = company_data.sort_values('Deal Date').index
            enhanced_data.loc[sorted_indices, 'Round_Sequence'] = range(1, len(sorted_indices) + 1)
        # Else if we have round maturity, sort by that
        elif 'Round_Maturity' in company_data.columns:
            sorted_indices = company_data.sort_values('Round_Maturity').index
            enhanced_data.loc[sorted_indices, 'Round_Sequence'] = range(1, len(sorted_indices) + 1)
    
    # Calculate round progression speed (average days between rounds)
    if 'Days_Since_Last_Funding' in enhanced_data.columns:
        for company in enhanced_data['Companies'].unique():
            company_data = enhanced_data[enhanced_data['Companies'] == company]
            if len(company_data) > 1:
                avg_days = company_data['Days_Since_Last_Funding'].mean()
                enhanced_data.loc[enhanced_data['Companies'] == company, 'Avg_Days_Between_Rounds'] = avg_days

    # 2.7 Add outlier indicators for extreme valuations
    print("- Adding outlier indicators for extreme valuations...")
    
    # Calculate z-score for valuations within each industry sector
    if 'Primary Industry Sector' in enhanced_data.columns:
        for sector in enhanced_data['Primary Industry Sector'].unique():
            if pd.notna(sector):
                sector_data = enhanced_data[enhanced_data['Primary Industry Sector'] == sector]
                
                if len(sector_data) > 5:  # Only calculate if we have enough samples
                    # Calculate z-score for this sector
                    sector_mean = sector_data[target_variable].mean()
                    sector_std = sector_data[target_variable].std()
                    
                    if sector_std > 0:  # Avoid division by zero
                        z_scores = (sector_data[target_variable] - sector_mean) / sector_std
                        enhanced_data.loc[sector_data.index, 'Valuation_Z_Score'] = z_scores
    
    # Flag outliers based on z-score
    if 'Valuation_Z_Score' in enhanced_data.columns:
        enhanced_data['Is_Valuation_Outlier'] = (
            (enhanced_data['Valuation_Z_Score'] > 3) | 
            (enhanced_data['Valuation_Z_Score'] < -3)
        ).astype(int)

    # Print the new features added
    new_features = [col for col in enhanced_data.columns if col not in valid_data.columns]
    print(f"\nAdded {len(new_features)} new features:")
    print(new_features)

    # Separate features into numerical and categorical
    numerical_features = []
    categorical_features = []

    # Get all features from the enhanced dataset
    all_features = list(enhanced_data.columns)
    # Remove the target variable from features list
    all_features.remove(target_variable) if target_variable in all_features else None

    for feature in all_features:
        if enhanced_data[feature].dtype in ['int64', 'float64']:
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    print(f"\nNumerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # 3. Create train/test split with IPOs in test set and ALL non-IPO rounds in train set
    print("\n3. Creating train-test split with IPOs in test set and ALL non-IPO rounds in training set...")

    # Check if necessary columns exist
    if 'Deal Type' in enhanced_data.columns:
        # Create feature matrix and target vector
        X = enhanced_data.copy()
        y = enhanced_data[target_variable]
        
        # Apply log transformation to the target variable to handle skewness
        y_log = np.log1p(y)
        
        # Split data into IPO and non-IPO deals
        ipo_mask = enhanced_data['Deal Type'] == "IPO"
        
        # Define train and test sets
        # ALL non-IPO deals go to train set, ALL IPO deals go to test set
        X_train = X[~ipo_mask].drop(columns=[target_variable])
        y_train = y_log[~ipo_mask]
        X_test = X[ipo_mask].drop(columns=[target_variable])
        y_test = y_log[ipo_mask]
        
        print(f"Training set (ALL non-IPO rounds): {len(X_train)} samples from {X_train['Companies'].nunique()} companies")
        print(f"Test set (ALL IPO rounds): {len(X_test)} samples")

        # Show distribution of VC rounds in train set
        if 'VC Round' in X_train.columns:
            vc_round_dist = X_train['VC Round'].value_counts(normalize=True) * 100
            print("\nDistribution of VC Rounds in training set:")
            for round_type, percentage in vc_round_dist.head(10).items():
                print(f"- {round_type}: {percentage:.1f}%")
        
        # 4. Feature preprocessing and model training
        print("\n4. Setting up feature preprocessing and model training pipelines...")
        
        # Define preprocessing steps for numerical and categorical features
        # Using RobustScaler for better handling of outliers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('interactions', InteractionFeatureTransformer(interaction_only=True, include_bias=False)),
            ('scaler', RobustScaler())  # Better handling of outliers than StandardScaler
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', CategoricalImputer(fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Dictionary of models to evaluate, including HuberRegressor for robust handling of outliers
        models = {
            'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42),
            'Huber Regressor': HuberRegressor(epsilon=1.35, max_iter=1000),  # More robust to outliers
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        # Create results DataFrame to store model performance
        results_df = pd.DataFrame(columns=['MAE', 'RMSE', 'R² Score', 'MAPE (%)'])
        
        print("Training and evaluating models...")
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create a pipeline with preprocessing and the model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Fit the pipeline on the training data
            try:
                pipeline.fit(X_train, y_train)
                
                # Make predictions on the test set
                y_pred_log = pipeline.predict(X_test)
                
                # Transform predictions back to original scale
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_test)
                
                # Calculate evaluation metrics with handling for infinite values
                # First, filter out any infinite values
                valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
                y_true_valid = y_true[valid_indices]
                y_pred_valid = y_pred[valid_indices]
                
                if len(y_true_valid) > 0:
                    mae = mean_absolute_error(y_true_valid, y_pred_valid)
                    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                    r2 = r2_score(y_true_valid, y_pred_valid)
                    mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
                    
                    if len(y_true) - len(y_true_valid) > 0:
                        print(f"Note: {len(y_true) - len(y_true_valid)} samples were filtered out due to infinite values")
                else:
                    print("Warning: All samples have infinite values, metrics will be set to NaN")
                    mae = rmse = r2 = mape = np.nan
                
                # Store results
                results_df.loc[name] = [mae, rmse, r2, mape]
                
                print(f"{name} Results:")
                print(f"  MAE: ${mae:,.2f}")
                print(f"  RMSE: ${rmse:,.2f}")
                print(f"  R² Score: {r2:.4f}")
                print(f"  MAPE: {mape:.2f}%")
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results_df.loc[name] = [np.nan, np.nan, np.nan, np.nan]
        
        # Sort models by MAPE (lower is better)
        results_df = results_df.sort_values('MAPE (%)')
        
        # Get the best model
        best_model_name = results_df.index[0]
        best_model_mape = results_df.loc[best_model_name, 'MAPE (%)'] 
        
        print(f"\nBest model: {best_model_name} with MAPE: {best_model_mape:.2f}%")
        
        # Retrain the best model on the entire dataset
        print(f"\nRetraining {best_model_name} on all training data for final model...")
        
        best_model = models[best_model_name]
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate the final model on the test set (IPO data)
        print("\nEvaluating final model on IPO test data...")
        y_pred_log = final_pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        # Calculate final evaluation metrics with handling for infinite values
        valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]
        
        if len(y_true_valid) > 0:
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            r2 = r2_score(y_true_valid, y_pred_valid)
            mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
            
            if len(y_true) - len(y_true_valid) > 0:
                print(f"Note: {len(y_true) - len(y_true_valid)} samples were filtered out due to infinite values")
        else:
            print("Warning: All samples have infinite values, metrics will be set to NaN")
            mae = rmse = r2 = mape = np.nan
        
        print(f"Final Model Performance on IPO Data:")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # 5. Analysis of predictions by funding round type
        print("\n5. Analyzing predictions by funding round type...")
        
        # Create dataframe with predictions and actual values
        pred_df = pd.DataFrame({
            'Company': enhanced_data.loc[X_test.index, 'Companies'],
            'Actual': y_true_valid,
            'Predicted': y_pred_valid,
            'Absolute Error': np.abs(y_true_valid - y_pred_valid),
            'Percentage Error': np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
        })
        
        # Add relevant features from the original dataset
        for col in ['VC Round', 'Is_Early_Round', 'Primary Industry Sector', 'Company_Had_Early_Round']:
            if col in enhanced_data.columns:
                pred_df[col] = enhanced_data.loc[X_test.index[valid_indices], col].values
        
        # Analyze error by whether the company had early rounds
        if 'Company_Had_Early_Round' in pred_df.columns:
            had_early_mask = pred_df['Company_Had_Early_Round'] == 1
            no_early_mask = ~had_early_mask
            
            # Calculate MAPE for companies with and without early rounds
            if had_early_mask.any():
                mape_with_early = pred_df[had_early_mask]['Percentage Error'].mean()
                count_with_early = had_early_mask.sum()
                
                print(f"\nCompanies with early rounds: {count_with_early}, MAPE: {mape_with_early:.2f}%")
            else:
                mape_with_early = np.nan
                count_with_early = 0
            
            if no_early_mask.any():
                mape_without_early = pred_df[no_early_mask]['Percentage Error'].mean()
                count_without_early = no_early_mask.sum()
                
                print(f"Companies without early rounds: {count_without_early}, MAPE: {mape_without_early:.2f}%")
            else:
                mape_without_early = np.nan
                count_without_early = 0
            
            # Create a bar chart to visualize the difference if both groups exist
            if count_with_early > 0 and count_without_early > 0:
                plt.figure(figsize=(10, 6))
                error_comparison = [mape_with_early, mape_without_early]
                counts = [count_with_early, count_without_early]
                labels = [f'With Early Rounds\n(n={count_with_early})', f'Without Early Rounds\n(n={count_without_early})']
                
                plt.bar(labels, error_comparison, color=['#3498db', '#e74c3c'])
                plt.title('MAPE by Early Round History')
                plt.ylabel('Mean Absolute Percentage Error (%)')
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on the bars
                for i, v in enumerate(error_comparison):
                    plt.text(i, v + 5, f'{v:.1f}%', ha='center')
                    
                plt.tight_layout()
                plt.savefig('approach4_early_round_mape_comparison.png')
                plt.show()
        
        # Create a scatter plot of actual vs. predicted values for visualization
        plt.figure(figsize=(12, 8))
        
        # Add a perfect prediction line
        max_val = max(max(y_true_valid), max(y_pred_valid))
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        # Plot the actual predictions, color by whether the company had early rounds
        if 'Company_Had_Early_Round' in pred_df.columns:
            # Plot points with early rounds
            early_mask = pred_df['Company_Had_Early_Round'] == 1
            plt.scatter(pred_df[early_mask]['Actual'], pred_df[early_mask]['Predicted'], 
                       alpha=0.6, label='Companies with Early Rounds', color='#3498db')
            
            # Plot points without early rounds
            plt.scatter(pred_df[~early_mask]['Actual'], pred_df[~early_mask]['Predicted'], 
                       alpha=0.6, label='Companies without Early Rounds', color='#e74c3c')
        else:
            # Plot all points the same color
            plt.scatter(y_true_valid, y_pred_valid, alpha=0.6)
            
        plt.title(f'Actual vs. Predicted IPO Valuations - Enhanced {best_model_name} Model')
        plt.xlabel('Actual Valuation ($)')
        plt.ylabel('Predicted Valuation ($)')
        
        # Use log scale for better visualization
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig('approach4_actual_vs_predicted.png')
        plt.show()
        
        # 6. Feature Importance Analysis
        print("\n6. Analyzing feature importance...")
        
        # Check if the model supports feature importances
        if hasattr(final_pipeline['model'], 'feature_importances_'):
            try:
                # Get feature names from preprocessor
                feature_names = []
                
                # Get numerical feature names (including interactions)
                numerical_features_out = final_pipeline['preprocessor'].transformers_[0][1]['interactions'].feature_names_out
                feature_names.extend(numerical_features_out)
                
                # Get categorical feature names
                categorical_features_out = final_pipeline['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
                feature_names.extend(categorical_features_out)
                
                # Get feature importances
                importances = final_pipeline['model'].feature_importances_
                
                # Ensure lengths match
                if len(importances) == len(feature_names):
                    # Create feature importance dataframe
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    })
                    
                    # Sort by importance
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    # Print top features
                    print("\nTop 20 most important features:")
                    print(feature_importance.head(20))
                    
                    # Highlight new features in the top features
                    new_feature_prefixes = [f.split('_x_')[0] for f in new_features if '_x_' in f] + new_features
                    top_new_features = []
                    
                    for feature in feature_importance['Feature'].head(20):
                        for prefix in new_feature_prefixes:
                            if str(feature).startswith(prefix) or str(feature).endswith(prefix):
                                top_new_features.append(feature)
                                break
                    
                    print(f"\nNew features among top 20: {len(top_new_features)}")
                    for f in top_new_features:
                        print(f"- {f}")
                    
                    # Plot top features
                    plt.figure(figsize=(14, 10))
                    plt.barh(feature_importance['Feature'].head(20)[::-1], 
                            feature_importance['Importance'].head(20)[::-1])
                    plt.title(f'Top 20 Feature Importances - {best_model_name}')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    plt.savefig('approach4_feature_importance.png')
                    plt.show()
                else:
                    print(f"Warning: Feature importances length ({len(importances)}) doesn't match feature names length ({len(feature_names)}).")
            except Exception as e:
                print(f"Error extracting feature importances: {e}")
        else:
            print(f"Model {best_model_name} doesn't support feature importance extraction.")
        
        # 7. Save the final model and feature information
        print("\n7. Saving the enhanced model...")
        
        model_dir = "saved_approach4_model"
        os.makedirs(model_dir, exist_ok=True)
        
        model_filename = os.path.join(model_dir, "approach4_valuation_prediction_model.pkl")
        feature_filename = os.path.join(model_dir, "approach4_model_features.pkl")
        
        # Save the trained model
        with open(model_filename, 'wb') as f:
            pickle.dump(final_pipeline, f)
            
        # Save feature information for future use
        feature_info = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'target_variable': target_variable,
            'new_features': new_features
        }
        
        with open(feature_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        
        print(f"\nEnhanced Approach 4 model saved to {model_filename}")
        print(f"Feature information saved to {feature_filename}")
        
        # 8. Create a prediction function for new companies
        print("\n8. Creating prediction function for new company data...")
        
        def predict_ipo_valuation(new_data):
            """Predict IPO valuation for a new company using the enhanced model.
            
            Parameters:
            -----------
            new_data : pandas DataFrame
                DataFrame containing company data. Should include basic fields like
                Deal Size, Pre-money Valuation, Industry, etc.
            
            Returns:
            --------
            numpy array
                Predicted IPO valuation(s) in the original scale (not log-transformed)
            """
            # Load the model and feature info
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            
            with open(feature_filename, 'rb') as f:
                feature_info = pickle.load(f)
            
            # Get list of required features
            required_features = feature_info['numerical_features'] + feature_info['categorical_features']
            
            # Create a copy of input data
            data = new_data.copy()
            
            # Handle missing required features
            missing_features = [f for f in required_features if f not in data.columns]
            for feature in missing_features:
                data[feature] = np.nan
            
            # Keep only required features
            try:
                data = data[required_features]
            except KeyError as e:
                print(f"Error with features: {e}")
                missing_cols = [col for col in required_features if col not in data.columns]
                print(f"Missing columns: {missing_cols}")
                return None
            
            # Make prediction (log-transformed)
            y_pred_log = model.predict(data)
            
            # Transform back to original scale
            y_pred = np.expm1(y_pred_log)
            
            return y_pred
        
        # Example prediction
        print("\n9. Example prediction:")
        try:
            sample_company = X_test.iloc[[0]].copy()
            predicted_val = predict_ipo_valuation(sample_company)
            actual_val = np.expm1(y_test.iloc[0])
            
            print(f"Sample company: {enhanced_data.loc[X_test.index[0], 'Companies']}")
            if 'VC Round' in enhanced_data.columns:
                print(f"VC Round: {enhanced_data.loc[X_test.index[0], 'VC Round']}")
            print(f"Predicted IPO valuation: ${predicted_val[0]:,.2f}")
            print(f"Actual IPO valuation: ${actual_val:,.2f}")
            print(f"Error: {abs(predicted_val[0] - actual_val) / actual_val * 100:.2f}%")
        except Exception as e:
            print(f"Error in example prediction: {e}")
        
        print("\n10. Summary of enhancements in Approach 4:")
        print("----------------------------------------")
        print("✓ Used ALL non-IPO funding rounds in the training set (instead of just the latest)")
        print("✓ Added funding round maturity indicators to better capture early rounds")
        print("✓ Added time-based features including company age and days between rounds")
        print("✓ Added growth rate features between funding rounds")
        print("✓ Added market condition features based on industry trends")
        print("✓ Created interaction features specifically to address early round valuations")
        print("✓ Added company sequential round features to track funding progression")
        print("✓ Added outlier detection and handling for extreme valuations")
        print("✓ Used robust scaling and metrics to handle outliers better")
        print("----------------------------------------")
        print(f"Best model: {best_model_name}")
        print(f"Performance: MAPE = {best_model_mape:.2f}%")
        print(f"Model saved to: {model_filename}")
        
        return final_pipeline, feature_info
        
    else:
        missing_cols = []
        if 'Deal Type' not in enhanced_data.columns:
            missing_cols.append('Deal Type')
        print(f"Warning: {', '.join(missing_cols)} column(s) not found. Cannot create the train-test split.")
        return None, None

if __name__ == "__main__":
    enhance_approach4()
