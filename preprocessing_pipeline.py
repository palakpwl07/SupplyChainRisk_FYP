
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Replace with actual dataset path

# 1. Drop rows with missing supplier_country_code
df = df.dropna(subset=['supplier_country_code'])

# 2. Missing Value Analysis (logged internally or manually observed)

# 3. Mean imputation for numerical features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 4. Label Encoding for Categorical Variables
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. Standardization of Numeric Variables
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Retain All Features (Initial Pass) - Pruning deferred

# 7. Train-Test Split (80:20, Stratified if target is categorical)
target_column = 'delivery_time_deviation'
X = df.drop(columns=[target_column])
y = df[target_column]

# Use stratification only if target is categorical or has few unique values
stratify = y if y.nunique() < 20 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# Final Outputs
print("Preprocessing Complete!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
