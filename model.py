import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
salary_data = pd.read_csv('data/ds_salaries.csv')

# Select relevant features
salary_data = salary_data[['experience_level', 'employment_type', 'job_title', 'salary_in_usd', 'company_size']]

# Encode experience level
encoder = OrdinalEncoder(categories=[['EN', 'MI', 'SE', 'EX']])
salary_data['experience_level_encoded'] = encoder.fit_transform(salary_data[['experience_level']])

# Encode company size
encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])
salary_data['company_size_encoded'] = encoder.fit_transform(salary_data[['company_size']])

# Encode employment type and job title
salary_data = pd.get_dummies(salary_data, columns=['employment_type', 'job_title'], drop_first=True, dtype=int)

# Drop original columns
salary_data = salary_data.drop(columns=['experience_level', 'company_size'])

# Split features and target
X = salary_data.drop(columns='salary_in_usd')
y = salary_data['salary_in_usd']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.2, shuffle=True)

# Train model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Evaluate model
y_pred = regr.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2: %.2f" % r2_score(y_test, y_pred))

# Save model
joblib.dump(regr, 'lin_regress.sav')

print("Model trained and saved!")