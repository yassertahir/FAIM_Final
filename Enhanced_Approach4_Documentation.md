# Enhanced Approach 4: IPO Valuation Prediction Using All Non-IPO Funding Rounds

## Overview

Enhanced Approach 4 builds upon Enhanced Approach 3 with a significant methodology change: using **ALL non-IPO funding rounds** for training instead of just the latest round for each company. This approach maximizes the available training data while maintaining the key enhancement features that help bridge the valuation gap between early funding rounds and IPOs.

## Key Differences from Enhanced Approach 3

| Feature | Enhanced Approach 3 | Enhanced Approach 4 |
|---------|---------------------|---------------------|
| Training data | Latest non-IPO round only | ALL non-IPO rounds |
| Data volume | 1 record per company | Multiple records per company (2-3x more data) |
| Scaling method | StandardScaler | RobustScaler (better for outliers) |
| Error metrics | MAE, RMSE, MAPE, R² | Added MdAPE, Median AE (more robust) |
| Outlier handling | Basic | Advanced (OutlierClipper, quantile-based) |
| Model robustness | Standard | Enhanced with Huber Regressor |

## Why Use All Non-IPO Rounds?

1. **Larger training dataset**: More data points generally lead to better model learning
2. **Captures funding progression**: The model learns valuation patterns across different stages
3. **Better representation of early rounds**: More early-round data helps model the large jumps to IPO
4. **Company diversity**: Includes companies at various stages of development
5. **Round-to-round patterns**: Can identify patterns in how valuations evolve between rounds

## Enhancements Implemented

### 1. Data Handling Improvements

- **RobustScaler**: Less sensitive to outliers than StandardScaler
- **OutlierClipper**: Custom transformer that clips extreme values at specified quantiles
- **Ensemble methods**: Option to create ensemble predictions from multiple models
- **Improved error handling**: Better handling of infinite/NaN values in metrics

### 2. Additional Features

All features from Enhanced Approach 3, plus:

- **Round sequence features**:
  - `Round_Sequence`: Numeric sequence of funding rounds for each company
  - `Avg_Days_Between_Rounds`: Average speed of funding progression
  
- **Outlier detection features**:
  - `Valuation_Z_Score`: Standardized score for valuations within industry sectors
  - `Is_Valuation_Outlier`: Binary flag for statistical outliers

### 3. Robust Metrics

- **Median Absolute Percentage Error (MdAPE)**: Median of percentage errors, less affected by extreme values
- **Median Absolute Error (MedAE)**: Median of absolute errors in dollar terms

## Implementation Details

The core differences in implementation compared to Approach 3:

```python
# Enhanced Approach 3: Train on latest non-IPO round only
for company in companies:
    company_data = non_ipo_data[non_ipo_data['Companies'] == company].copy()
    latest_data = company_data.sort_values('Deal Date', ascending=False).iloc[0:1]
    train_indices.extend(latest_data.index)

# Enhanced Approach 4: Train on ALL non-IPO rounds
# Define train and test sets - ALL non-IPO goes to train, ALL IPO goes to test
X_train = X[~ipo_mask].drop(columns=[target_variable])
y_train = y_log[~ipo_mask]
X_test = X[ipo_mask].drop(columns=[target_variable])
y_test = y_log[ipo_mask]
```

## Model Pipeline

The Enhanced Approach 4 model pipeline includes:

1. **Preprocessing**:
   - Median imputation for missing numerical values
   - Outlier clipping at 1st and 99th percentiles
   - Feature interactions between numerical features
   - Robust scaling for numerical features
   - One-hot encoding for categorical features

2. **Model options**:
   - XGBoost
   - Random Forest
   - Gradient Boosting
   - Extra Trees
   - Huber Regressor (robust to outliers)
   - Ridge Regression
   - Ensemble (average of all models)

## Performance Analysis

The model evaluates performance specifically on:

- Overall accuracy metrics (MAE, RMSE, R², MAPE, MdAPE)
- Performance comparison between companies with and without early round history
- Analysis by industry sector
- Error distribution visualization
- Actual vs. predicted scatter plots with log scaling

## Potential Limitations

- Using all rounds may introduce dependencies between training samples from the same company
- May require more computational resources due to larger training set
- Requires careful handling of sequential data aspects

## Comparison to Other Approaches

| Approach | Training Data | Test Data | Key Advantage |
|----------|--------------|-----------|---------------|
| Approach 1 | All non-IPO rounds | All IPO rounds | Maximum training data |
| Approach 2 | 75% of IPO rounds | 25% of IPO rounds | Focus on IPO data only |
| Approach 3 | Latest non-IPO round per company | All IPO rounds | Better company matching |
| Enhanced Approach 3 | Latest non-IPO round with enhanced features | All IPO rounds | Handles valuation jumps |
| Enhanced Approach 4 | ALL non-IPO rounds with enhanced features | All IPO rounds | Maximum data + robustness |
