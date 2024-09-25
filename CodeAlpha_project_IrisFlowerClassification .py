#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset from a CSV file
iris_df = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
iris_df.head()


# In[3]:


#check missing value 
print(iris_df.isnull().sum())


# In[4]:


#check data type of the column 
print(iris_df.dtypes)


# In[7]:


#preprocessing the dataset 
from sklearn.preprocessing import LabelEncoder 

#initialize label encoder 

le = LabelEncoder()
iris_df['Species'] = le.fit_transform(iris_df['Species'])

#check transformed values 
iris_df.head()


# In[8]:


#Splitting the data into features and Labels
#features(independent variables)
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

#label (dependet variable)
y = iris_df['Species']


# In[9]:


#train-test split 
from sklearn.model_selection import train_test_split 

#Split the data into training and testing sets 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[12]:


#training model 
#1.Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the RandomForestClassifier
model_rf = RandomForestClassifier()

# Train the model
model_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")


# In[17]:


#2.logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=200)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_log = log_reg.predict(X_test)

# Evaluate the model's performance
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {accuracy_log * 100:.2f}%")


# In[18]:


#3.K-Nearest Neighbors(KNN)
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Evaluate the model's performance
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")


# In[20]:


#Evaluate model 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)  # Use predictions from any model

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[21]:


print(f"Logistic Regression Accuracy: {accuracy_log * 100:.2f}%")
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

