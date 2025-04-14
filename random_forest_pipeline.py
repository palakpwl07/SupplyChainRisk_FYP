"""
Random Forest Regressor Pipeline
Author: Palak Porwal
Last Modified: April 2025

This script defines a comprehensive machine learning pipeline using Random Forest
for regression tasks. It includes:
- Advanced preprocessing (scaling, feature selection)
- Hyperparameter optimization with GridSearchCV
- Integrated performance reporting
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add interaction terms or domain-specific features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if 'inventory_days' in X_.columns and 'lead_time' in X_.columns:
            X_['inventory_buffer_ratio'] = X_['inventory_days'] / (X_['lead_time'] + 1)
        return X_


def build_random_forest_model(X_train, y_train):
    logger.info("▶ Initializing pipeline for Random Forest Regressor...")

    # Step 1: Preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(score_func=f_regression, k='all')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('pca', PCA(n_components=0.95, random_state=42))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    # Step 2: Complete pipeline with custom feature engineering
    pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Step 3: Define hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [10, 15, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__max_features': ['sqrt', 'log2', None],
        'regressor__bootstrap': [True, False]
    }

    logger.info("▶ Performing grid search with cross-validation...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='neg_root_mean_squared_error',
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"✅ Best parameters found: {grid_search.best_params_}")
    logger.info("✅ Best RMSE Score: %.4f", -grid_search.best_score_)

    # Step 4: Optional cross-validation on the best estimator
    logger.info("▶ Cross-validating final model...")
    best_model = grid_search.best_estimator_
    cv_rmse = -cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    logger.info("✅ Cross-validated RMSE scores: %s", np.round(cv_rmse, 4))
    logger.info("✅ Mean RMSE: %.4f", np.mean(cv_rmse))

    return best_model
