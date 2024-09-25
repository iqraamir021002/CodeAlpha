#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:



# Step 1: Load the dataset

# Assuming the dataset is named 'sales_data.csv'
df = pd.read_csv('Advertising.csv')
df


# In[5]:



# Step 2: Data Preprocessing
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Drop any rows with missing values (if necessary)
df.dropna(inplace=True)

# Check if there are any columns with missing values or other issues
print(df.info())

# Extract feature columns and target variable
X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']  # Target variable


# In[ ]:


# Step 3: Scaling the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[7]:


# Step 5: Train Multiple Models

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


# In[8]:


# Step 6: Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name}:")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    print("\n")

# Evaluate each model
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_dt, "Decision Tree")


# In[9]:



# Step 7: Visualization
# Visualization: Actual vs Predicted Sales for Random Forest Model
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.show()

