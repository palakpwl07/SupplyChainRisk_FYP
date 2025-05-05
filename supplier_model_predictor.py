
import pandas as pd
import numpy as np
import os
import joblib
import logging
from collections import defaultdict

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_supplier_risk_model(data_path='data/supplier_data_cleaned.csv', model_dir='models/'):
    os.makedirs(model_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(model_dir, "training.log")), logging.StreamHandler()]
    )
    
    df = pd.read_csv(data_path)
    TARGET_COL = "risk_classification"
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_labels = dict(zip(le.classes_, le.transform(le.classes_)))
    logging.info(f"Encoded classes: {class_labels}")
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    logreg = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.85, random_state=42)
    
    selector = SelectFromModel(estimator=rf, threshold="median")
    
    meta_params = {
        'penalty': ['l2'],
        'C': [0.1, 0.5, 1.0, 2.0],
        'solver': ['liblinear']
    }
    
    final_estimator = RandomizedSearchCV(
        estimator=LogisticRegression(class_weight='balanced'),
        param_distributions=meta_params,
        cv=3,
        n_iter=5,
        random_state=42,
        n_jobs=-1
    )
    
    stack_model = StackingClassifier(
        estimators=[('lr', logreg), ('rf', rf), ('gb', gb)],
        final_estimator=final_estimator,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('feature_selector', selector),
        ('classifier', stack_model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    full_pipeline.fit(X_train, y_train)
    
    y_pred = full_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    acc = accuracy_score(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, full_pipeline.predict_proba(X_test), multi_class='ovr')
    except:
        roc_auc = "NA"
    
    logging.info("\n" + report)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"ROC-AUC (OvR): {roc_auc}")
    
    importances = defaultdict(float)
    try:
        feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
        selector_mask = full_pipeline.named_steps['feature_selector'].get_support()
        selected_features = np.array(feature_names)[selector_mask]
        base_rf = full_pipeline.named_steps['classifier'].estimators_[1]
        for name, score in zip(selected_features, base_rf.feature_importances_):
            importances[name] += score

        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        logging.info("Top 10 important features:")
        for feat, score in top_features:
            logging.info(f"{feat}: {score:.4f}")
    except Exception as e:
        logging.warning("Failed to extract feature importances: %s", e)
    
    joblib.dump(full_pipeline, os.path.join(model_dir, "ensemble_supplier_model_advanced.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "input_columns.pkl"))
    logging.info("Model and artifacts saved to /models/")
