# Extract and Save Models from Notebook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import joblib
import urllib.request
import os

print("🚀 Starting model extraction...")

# Download Adult dataset if not present
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
if not os.path.exists('adult.data'):
    print("📥 Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, 'adult.data')

# Load and prepare data
col_names = [
    'age','workclass','fnlwgt','education','education-num','marital-status','occupation',
    'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'
]

print("📊 Loading and processing data...")
raw = pd.read_csv('adult.data', names=col_names, na_values='?', skipinitialspace=True)

# Basic cleaning
data = raw.dropna().copy()

# Map income to binary and create regression target
data['income_binary'] = data['income'].map({'<=50K':0, '>50K':1})
salary_map = {'<=50K':35000, '>50K':65000}
data['salary_reg'] = data['income'].map(salary_map)

# Separate features and targets
X = data.drop(columns=['income','income_binary','salary_reg'])
y_reg = data['salary_reg']

# Identify categorical & numerical columns
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

print(f"✅ Found {len(cat_cols)} categorical and {len(num_cols)} numerical features")

# Create preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# Create regression pipeline
print("🤖 Training regression model...")
reg_pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('model', LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1  # Suppress training output
    ))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_pipe.fit(X_train, y_train)

# Test the model
test_pred = reg_pipe.predict(X_test)
print(f"✅ Model trained successfully!")
print(f"📈 Sample predictions: {test_pred[:5]}")

# Save the models
print("💾 Saving models...")
joblib.dump(reg_pipe, 'employee_salary_model.pkl')
print("✅ Main model saved as 'employee_salary_model.pkl'")

# Save just the preprocessor separately (for backup)
joblib.dump(preprocess, 'preprocessor.pkl') 
print("✅ Preprocessor saved as 'preprocessor.pkl'")

# Test loading
print("🧪 Testing model loading...")
try:
    loaded_model = joblib.load('employee_salary_model.pkl')
    test_prediction = loaded_model.predict(X_test[:1])
    print(f"✅ Model loads successfully! Test prediction: {test_prediction[0]}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("🎉 Model extraction complete!")
