import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and filter data for specific job titles
salary_data = pd.read_csv('data/ds_salaries.csv')
valid_titles = ['Data Engineer', 'Data Scientist', 'Machine Learning Engineer']
salary_data = salary_data[salary_data['job_title'].isin(valid_titles)]

# Select features
salary_data = salary_data[['experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'company_size']]

# Encode experience level
encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
salary_data['experience_level_encoded'] = encoder.fit_transform(salary_data[['experience_level']])

# Encode company size
encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
salary_data['company_size_encoded'] = encoder.fit_transform(salary_data[['company_size']])

# Create dummies
salary_data = pd.get_dummies(salary_data, columns=['employment_type', 'job_title'])

# Drop original columns
salary_data = salary_data.drop(columns=['experience_level', 'company_size'])

# Split features and target
X = salary_data.drop(columns='salary_in_usd')
y = salary_data['salary_in_usd']

print("Feature columns:", X.columns.tolist())

# Train model
regr = linear_model.LinearRegression()
regr.fit(X, y)

joblib.dump(regr, 'lin_regress.sav')
print("Model trained and saved!")
