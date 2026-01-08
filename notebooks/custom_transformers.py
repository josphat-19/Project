from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for initial data cleaning, type handling, imputation, 
    and outlier management, designed for the first step of the pipeline.
    """
    def __init__(self, id_col='CustomerID', num_features=None, cat_features=None, 
                 q_lower=0.05, q_upper=0.95):
        
        self.id_col = id_col
        self.num_features = num_features if num_features is not None else []
        self.cat_features = cat_features if cat_features is not None else []
        self.q_lower = q_lower
        self.q_upper = q_upper
        
        # Parameters learned during fit
        self.imputation_values = {}
        self.capping_limits = {} 
        self.cat_imputation_value = 'Missing' 
        
    def _coerce_types(self, X):
        """Helper to ensure categorical features are strings and numerical are floats."""
        X_copy = X.copy()
        
        # 1. Type Correction: Handle columns read as objects but meant to be numeric (if applicable)
        # Note: If your dataset had TotalCharges as object, you'd handle it here.
        # For this dataset, all numerics are floats/ints.
        
        # 2. Ensure numerical features are float types for calculation
        for col in self.num_features:
            if col in X_copy.columns:
                 # Convert to numeric, errors='coerce' turns non-numeric values into NaN
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

        # 3. Ensure categorical features are string types for consistent handling
        for col in self.cat_features:
            if col in X_copy.columns:
                 X_copy[col] = X_copy[col].astype(str)
                 
        return X_copy

    def fit(self, X, y=None):
        """
        Learns imputation values (median) and outlier limits (quantiles) 
        from the training data.
        """
        X_processed = self._coerce_types(X)
        
        # --- 1. Learn Imputation Values (Median) ---
        for col in self.num_features:
            if col in X_processed.columns:
                # Learn median for numerical imputation (robust to outliers)
                self.imputation_values[col] = X_processed[col].median()
        
        # --- 2. Learn Outlier Capping Limits (Quantiles) ---
        for col in self.num_features:
            if col in X_processed.columns:
                # Calculate the 5th and 95th percentiles from the training data
                lower_bound = X_processed[col].quantile(self.q_lower)
                upper_bound = X_processed[col].quantile(self.q_upper)
                self.capping_limits[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X, y=None):
        """
        Applies cleaning, imputation, outlier management, and drops ID column.
        """
        X_transformed = self._coerce_types(X)

        # 1. Drop Irrelevant Columns
        if self.id_col in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=[self.id_col], errors='ignore')

        # 2. Handle Missing Values (Imputation)
        
        # a) Numerical Imputation (using the median learned in fit)
        for col, value in self.imputation_values.items():
            if col in X_transformed.columns:
                 X_transformed[col] = X_transformed[col].fillna(value)

        # b) Categorical Imputation (filling with 'Missing')
        for col in self.cat_features:
            if col in X_transformed.columns:
                 X_transformed[col] = X_transformed[col].fillna(self.cat_imputation_value)
                 
        # 3. Manage Outliers (Capping/Clamping)
        for col in self.num_features:
            if col in X_transformed.columns and col in self.capping_limits:
                lower, upper = self.capping_limits[col]
                # Apply capping: values below 'lower' become 'lower', values above 'upper' become 'upper'
                X_transformed[col] = np.clip(X_transformed[col], lower, upper)
                
        return X_transformed

class CategoryGrouper(BaseEstimator, TransformerMixin): 
    def __init__(self, mappings):
        self.mappings = mappings
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(mapping)
        return X_copy


# Custom transformer to add new features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for creating new, predictive features (feature engineering) 
    from existing columns.

    This class handles zero replacement for ratio denominators and computes 
    various interaction and rate features such as Customer Value Score, 
    Activity Density, and a custom Churn Risk Score.
    
    The new features are:
    - CustomerValueScore: A composite loyalty/monetary score.
    - ActivityDensity: Rate of ordering relative to the time since the last order.
    - AvgCashbackPerOrder: Monetary reward efficiency.
    - RecentActivity (Flag): Binary flag for high recent engagement.
    - Frequency: Average orders per unit of tenure.
    - ChurnRiskScore: A heuristic-based aggregate score of risk factors.
    """
    def __init__(self, numerical_features: list, categorical_features: list):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_names = []
    
    def fit(self, X, y=None):
        """
        The FeatureEngineer is a stateless transformer for predefined calculations.
        It does not learn any parameters from the data during the fit process.
        
        Args:
            X (pd.DataFrame): Input feature data.
            y (pd.Series, optional): Target variable. Defaults to None.
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the feature engineering steps.

        Args:
            X (pd.DataFrame): Input feature data (already cleaned and imputed).

        Returns:
            pd.DataFrame: DataFrame containing original features plus new engineered features.
        """
        X = X.copy()

        # --- 1. Safe Zero Handling for Ratio Calculations ---
        # NOTE: This step is crucial. Replacing 0 with a very small number (0.001) prevents 
        # ZeroDivisionError when calculating ratios like Frequency or ActivityDensity.
        # This is necessary even though the DataCleaner handled NaNs.
        X['Tenure_adj'] = X['Tenure'].replace(0, 0.001)
        X['DaySinceLastOrder_adj'] = X['DaySinceLastOrder'].replace(0, 0.001)
        X['OrderCount_adj'] = X['OrderCount'].replace(0, 0.001)
        
        # --- 2. Interaction and Composite Scores ---
        
        # Customer Value Score: A weighted sum of key monetary and loyalty indicators.
        X['CustomerValueScore'] = (X['CashbackAmount'] * 0.4 + 
                                   X['OrderCount_adj'] * 0.3 + 
                                   X['Tenure_adj'] * 0.3)
        
        # Activity Density: Rate of ordering relative to the time since the last order.
        X['ActivityDensity'] = X['OrderCount_adj'] / X['DaySinceLastOrder_adj']

        # Avg Cashback Per Order: Monetary reward efficiency.
        X['AvgCashbackPerOrder'] = X['CashbackAmount'] / X['OrderCount_adj']
        
        # --- 3. Behavioral Flags and Rates ---
        
        # Recent Activity Flag: A binary flag for customers highly active in the last week.
        X['RecentActivity'] = (X['DaySinceLastOrder'] < 7).astype(int)
        
        # Average Orders Per Unit Tenure (Frequency): A measure of loyalty/engagement intensity.
        X['Frequency'] = X['OrderCount_adj'] / X['Tenure_adj']
        
        # Churn Risk Score: A composite score based on known risk factors (heuristics).
        # This combines binary flags for complaints, low satisfaction, low tenure, and inactivity.
        X['ChurnRiskScore'] = (
            (X['Complain'].astype(int) * 0.3) +
            ((X['DaySinceLastOrder'] > 14).astype(int) * 0.2) +
            ((X['SatisfactionScore'].astype(int) < 3).astype(int) * 0.2) +
            ((X['Tenure'] < 3).astype(int) * 0.3)
        )

        # --- 4. Cleanup ---
        # Drop the temporary adjusted columns used for safe division
        X = X.drop(['Tenure_adj', 'DaySinceLastOrder_adj', 'OrderCount_adj'], axis=1)
        # 1. Combine the two lists in the desired order
        ordered_cols = self.numerical_features + self.categorical_features
        
        # 2. Check if the DataFrame contains all required columns
        # (This is a good practice safeguard)
        missing_cols = set(ordered_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"FeatureEngineer is missing columns before reordering: {missing_cols}")

        # 3. Reorder the DataFrame using the fixed column list
        X = X[ordered_cols]
  
        
        # Store the final feature names for reference in subsequent pipeline steps
        self.feature_names = list(X.columns)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the columns output by the transformer, 
        ensuring scikit-learn compatibility.
        """
        return self.feature_names
