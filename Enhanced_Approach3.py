"""
Enhanced Approach 3 for IPO Valuation Prediction

This implementation enhances Approach 3 by adding features that account for large valuation jumps 
when the latest available funding round is an early round like angel investment.

Additional features include:
1. Time-based features (days since last funding round)
2. Funding round maturity indicators
3. Growth rate features between available funding rounds
4. Market condition features at the time of IPO
5. Interaction features between early rounds and company characteristics
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
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

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    # Also filter out infinite values
    if np.isfinite(y_true).all() and np.isfinite(y_pred).all():
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        # Replace infinite values with NaN and then drop them
        with np.errstate(all='ignore'):
            result = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        result = np.where(np.isfinite(result), result, np.nan)
        return np.nanmean(result) * 100

def enhance_approach3():
    """
    Enhanced implementation of Approach 3 with additional features to account for
    large valuation jumps between early funding rounds and IPO.
    """
    print("Enhanced Approach 3: Adding features to handle early-to-IPO valuation jumps")

    # Load the dataset
    print("\nLoading and preparing dataset...")
    try:
        df = pd.read_csv('combined_ipo_with_urls.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Number of unique companies: {df['Companies'].nunique()}")
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
    round_maturity_map = {
        'Angel': 1,
        'Seed': 2,
        'Series A': 3,
        'Series B': 4,
        'Series C': 5,
        'Series D': 6,
        'Series E': 7,
        'Series F': 8,
        'Series G': 9,
        'Series H': 10,
        'Series I': 11,
        'Late Stage': 12,
        'IPO': 13
    }

    # Create a new feature for funding round maturity
    if 'VC Round' in enhanced_data.columns:
        enhanced_data['Round_Maturity'] = enhanced_data['VC Round'].map(round_maturity_map).fillna(0)
        
        # Create a binary indicator for early rounds (Angel or Seed)
        enhanced_data['Is_Early_Round'] = enhanced_data['VC Round'].isin(['Angel', 'Seed']).astype(int)
        
        # Create a binary indicator for late stage rounds
        enhanced_data['Is_Late_Round'] = enhanced_data['VC Round'].isin(['Late Stage', 'Series D', 
                                                                        'Series E', 'Series F', 
                                                                        'Series G', 'Series H', 
                                                                        'Series I']).astype(int)
    else:
        print("Warning: 'VC Round' column not found. Skipping round maturity features.")

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

    # 3. Create train/test split with IPOs in test set and latest non-IPO rounds in train set
    print("\n3. Creating train-test split with IPOs in test set and latest non-IPO rounds in training set...")

    # Check if necessary columns exist
    if 'Deal Type' in enhanced_data.columns and 'Companies' in enhanced_data.columns:
        # Create feature matrix and target vector
        X = enhanced_data.copy()
        y = enhanced_data[target_variable]
        
        # Apply log transformation to the target variable to handle skewness
        y_log = np.log1p(y)
        
        # Split data into IPO and non-IPO deals
        ipo_mask = enhanced_data['Deal Type'] == "IPO"
        ipo_data = enhanced_data[ipo_mask].copy()
        non_ipo_data = enhanced_data[~ipo_mask].copy()
        
        print(f"IPO deals found: {len(ipo_data)} out of {len(enhanced_data)} total records")
        print(f"Non-IPO deals found: {len(non_ipo_data)} out of {len(enhanced_data)} total records")
        
        # Initialize train indices list
        train_indices = []
        
        # Get a list of all unique companies
        companies = non_ipo_data['Companies'].unique()
        print(f"Found {len(companies)} unique companies with non-IPO deals")
        
        # For each company in the non-IPO dataset, get the latest funding round
        for company in companies:
            company_data = non_ipo_data[non_ipo_data['Companies'] == company].copy()
            
            # Try to sort by date if available
            if 'Deal Date' in company_data.columns and pd.notna(company_data['Deal Date']).any():
                # Sort by date (most recent first) and take the first row
                latest_data = company_data.sort_values('Deal Date', ascending=False).iloc[0:1]
                train_indices.extend(latest_data.index)
            else:
                # If no date information, just take the first row
                train_indices.append(company_data.index[0])
        
        # All IPO data goes to test set
        test_indices = ipo_data.index.tolist()
        
        # Create train and test sets
        X_train = X.loc[train_indices]
        y_train = y_log.loc[train_indices]
        X_test = X.loc[test_indices]
        y_test = y_log.loc[test_indices]
        
        # Drop the target variable from feature matrix
        X_train = X_train.drop(columns=[target_variable])
        X_test = X_test.drop(columns=[target_variable])
        
        print(f"Training set (latest non-IPO rounds): {len(X_train)} samples")
        print(f"Test set (all IPO rounds): {len(X_test)} samples")
        
        # 4. Feature preprocessing and model training
        print("\n4. Setting up feature preprocessing and model training pipelines...")
        
        # Define preprocessing steps for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('interactions', InteractionFeatureTransformer(interaction_only=True, include_bias=False)),
            ('scaler', StandardScaler())
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
        
        # Dictionary of models to evaluate
        models = {
            'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42),
            # 'Ridge': Ridge(alpha=1.0, random_state=42)
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
            pipeline.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred_log = pipeline.predict(X_test)
            
            # Transform predictions back to original scale
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test)
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            
            # Store results
            results_df.loc[name] = [mae, rmse, r2, mape]
            
            print(f"{name} Results:")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
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
        
        # Calculate final evaluation metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        print(f"Final Model Performance on IPO Data:")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # 5. Analysis of predictions by funding round type
        print("\n5. Analyzing predictions by funding round type...")
        
        # Create dataframe with predictions and actual values
        pred_df = pd.DataFrame({
            'Company': enhanced_data.loc[test_indices, 'Companies'],
            'Actual': y_true,
            'Predicted': y_pred,
            'Absolute Error': np.abs(y_true - y_pred),
            'Percentage Error': np.abs((y_true - y_pred) / y_true) * 100
        })
        
        # Add relevant features from the original dataset
        for col in ['VC Round', 'Is_Early_Round', 'Primary Industry Sector', 'Company_Had_Early_Round']:
            if col in enhanced_data.columns:
                pred_df[col] = enhanced_data.loc[test_indices, col].values
        
        # Analyze error by whether the company had early rounds
        if 'Company_Had_Early_Round' in pred_df.columns:
            had_early_mask = pred_df['Company_Had_Early_Round'] == 1
            no_early_mask = ~had_early_mask
            
            # Calculate MAPE for companies with and without early rounds
            mape_with_early = pred_df[had_early_mask]['Percentage Error'].mean()
            mape_without_early = pred_df[no_early_mask]['Percentage Error'].mean() if no_early_mask.any() else np.nan
            
            count_with_early = had_early_mask.sum()
            count_without_early = no_early_mask.sum()
            
            print(f"\nError analysis for companies with vs without early rounds:")
            print(f"Companies with early rounds: {count_with_early}, MAPE: {mape_with_early:.2f}%")
            print(f"Companies without early rounds: {count_without_early}, MAPE: {mape_without_early:.2f}%")
            
            # Create a bar chart to visualize the difference
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
            plt.show()
        
        # Create a scatter plot of actual vs. predicted values for visualization
        plt.figure(figsize=(12, 8))
        
        # Add a perfect prediction line
        max_val = max(max(y_true), max(y_pred))
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
            plt.scatter(y_true, y_pred, alpha=0.6)
            
        plt.title(f'Actual vs. Predicted IPO Valuations - Enhanced {best_model_name} Model')
        plt.xlabel('Actual Valuation ($)')
        plt.ylabel('Predicted Valuation ($)')
        
        # Use log scale for better visualization
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
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
                    plt.show()
                else:
                    print(f"Warning: Feature importances length ({len(importances)}) doesn't match feature names length ({len(feature_names)}).")
            except Exception as e:
                print(f"Error extracting feature importances: {e}")
        else:
            print(f"Model {best_model_name} doesn't support feature importance extraction.")
        
        # 7. Save the final model and feature information
        print("\n7. Saving the enhanced model...")
        
        model_dir = "saved_enhanced_ipo_model"
        os.makedirs(model_dir, exist_ok=True)
        
        model_filename = os.path.join(model_dir, "enhanced_valuation_prediction_model.pkl")
        feature_filename = os.path.join(model_dir, "enhanced_model_features.pkl")
        
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
        
        print(f"\nEnhanced model saved to {model_filename}")
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
            
            # Handle feature engineering for the new data
            # Add round maturity if VC Round is available
            if 'VC Round' in data.columns:
                round_maturity_map = {
                    'Angel': 1, 'Seed': 2, 'Series A': 3, 'Series B': 4,
                    'Series C': 5, 'Series D': 6, 'Series E': 7, 'Series F': 8,
                    'Series G': 9, 'Series H': 10, 'Series I': 11, 'Late Stage': 12,
                    'IPO': 13
                }
                data['Round_Maturity'] = data['VC Round'].map(round_maturity_map).fillna(0)
                data['Is_Early_Round'] = data['VC Round'].isin(['Angel', 'Seed']).astype(int)
                data['Is_Late_Round'] = data['VC Round'].isin(['Late Stage', 'Series D', 
                                                            'Series E', 'Series F', 
                                                            'Series G', 'Series H', 
                                                            'Series I']).astype(int)
            
            # Add company age if Year Founded and Deal Date are available
            if 'Year Founded' in data.columns and 'Deal Date' in data.columns:
                data['Deal Date'] = pd.to_datetime(data['Deal Date'], errors='coerce')
                data['Company_Age_at_Deal'] = data['Deal Date'].dt.year - data['Year Founded']
                data.loc[data['Company_Age_at_Deal'] < 0, 'Company_Age_at_Deal'] = np.nan
                data.loc[data['Company_Age_at_Deal'] > 100, 'Company_Age_at_Deal'] = np.nan
                
                # Add deal year and quarter
                data['Deal_Year'] = data['Deal Date'].dt.year
                data['Deal_Quarter'] = data['Deal Date'].dt.quarter
            
            # Create interaction features if possible
            if 'Is_Early_Round' in data.columns and 'Company_Age_at_Deal' in data.columns:
                data['Early_Round_x_Age'] = data['Is_Early_Round'] * data['Company_Age_at_Deal']
            
            if 'Is_Early_Round' in data.columns and 'Deal Size' in data.columns:
                data['Early_Round_x_Deal_Size'] = data['Is_Early_Round'] * data['Deal Size']
            
            # For new data, we don't have IPO flag information typically
            data['Is_IPO'] = 0  # Default to non-IPO for new prediction data
            data['Company_Had_Early_Round'] = data['Is_Early_Round'] if 'Is_Early_Round' in data.columns else 0
            data['IPO_and_Had_Early_Round'] = 0  # Default value
            
            # Fill missing required features with NaN
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
            
            print(f"Sample company: {enhanced_data.loc[test_indices[0], 'Companies']}")
            if 'VC Round' in enhanced_data.columns:
                print(f"VC Round: {enhanced_data.loc[test_indices[0], 'VC Round']}")
            print(f"Predicted IPO valuation: ${predicted_val[0]:,.2f}")
            print(f"Actual IPO valuation: ${actual_val:,.2f}")
            print(f"Error: {abs(predicted_val[0] - actual_val) / actual_val * 100:.2f}%")
        except Exception as e:
            print(f"Error in example prediction: {e}")
        
        print("\n10. Summary of enhancements:")
        print("----------------------------")
        print("✓ Added funding round maturity indicators to better capture early rounds")
        print("✓ Added time-based features including company age and days between rounds")
        print("✓ Added growth rate features between funding rounds")
        print("✓ Added market condition features based on industry trends")
        print("✓ Created interaction features specifically to address early round valuations")
        print("✓ Improved model training with more relevant features for valuation prediction")
        print("----------------------------")
        print(f"Best model: {best_model_name}")
        print(f"Performance: MAPE = {best_model_mape:.2f}%")
        print(f"Model saved to: {model_filename}")
        
        return final_pipeline, feature_info
        
    else:
        missing_cols = []
        if 'Deal Type' not in enhanced_data.columns:
            missing_cols.append('Deal Type')
        if 'Companies' not in enhanced_data.columns:
            missing_cols.append('Companies')
        print(f"Warning: {', '.join(missing_cols)} column(s) not found. Cannot create the train-test split.")
        return None, None

if __name__ == "__main__":
    enhance_approach3()
