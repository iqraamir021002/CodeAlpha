#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset from the file 'car data.csv'
df = pd.read_csv('car data.csv')
df


# In[6]:



df = pd.read_csv('car data.csv')

# Data Preprocessing
df['Age'] = 2024 - df['Year']  # Create an 'Age' column instead of using 'Year'

# Drop 'Car_Name' as it may not have a significant impact
df.drop('Car_Name', axis=1, inplace=True)

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])      # Convert 'Fuel_Type' to numeric
df['Selling_type'] = le.fit_transform(df['Selling_type']) # Convert 'Selling_type' to numeric
df['Transmission'] = le.fit_transform(df['Transmission']) # Convert 'Transmission' to numeric

# Drop the 'Year' column after calculating age
df.drop('Year', axis=1, inplace=True)

# Split the data into features (X) and target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Check for any missing values (handle them if present)
print("Missing values in the dataset:", X.isnull().sum().sum())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[7]:



# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 3. Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)


# In[10]:


# Evaluate models
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("===============================")
    print(f"{model_name}:")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    print("===============================")
    print("\n")

# Print the evaluation metrics for each model
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_dt, "Decision Tree")

# Visualization: Actual vs Predicted for Random Forest (you can do for other models similarly)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Selling Prices')
plt.ylabel('Predicted Selling Prices')
plt.title('Actual vs Predicted Car Prices (Random Forest)')
plt.show()


# In[ ]:





# In[ ]:




