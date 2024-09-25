#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets (adjust paths to your files)
df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# In[4]:


df1


# In[5]:


df2 


# In[7]:



# Convert 'Date' column to datetime format for both datasets
df1[' Date'] = pd.to_datetime(df1[' Date'], errors='coerce')
df2[' Date'] = pd.to_datetime(df2[' Date'], errors='coerce')


# In[8]:


# Let's check for missing values in key columns (Unemployment Rate, Date, Region)
print(df1.isnull().sum())
print(df2.isnull().sum())


# In[10]:


# Fill or drop missing values (you can choose to fill with mean/median or drop rows with missing data)
df1.dropna(subset=[' Estimated Unemployment Rate (%)'], inplace=True)
df2.dropna(subset=[' Estimated Unemployment Rate (%)'], inplace=True)


# In[14]:


# Focus on unemployment rate analysis. Let’s aggregate by Date and Region (if needed)
# Here we aggregate by date for a global unemployment trend
df1_grouped = df1.groupby(' Date')[' Estimated Unemployment Rate (%)'].mean().reset_index()
df2_grouped = df2.groupby(' Date')[' Estimated Unemployment Rate (%)'].mean().reset_index()


# In[16]:


# Plot the unemployment rate trends over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=' Date', y=' Estimated Unemployment Rate (%)', data=df1_grouped, label='Dataset 1 - India')
sns.lineplot(x=' Date', y=' Estimated Unemployment Rate (%)', data=df2_grouped, label='Dataset 2')
plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', label='COVID-19 Start')
plt.title('Unemployment Rate Trends (Pre & Post COVID-19)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


# In[19]:


# Optional: zoom into the COVID-19 period to observe the impact in more detail
covid_period = df1_grouped[df1_grouped[' Date'] >= '2020-03-01']

plt.figure(figsize=(10, 5))
sns.lineplot(x=' Date', y=' Estimated Unemployment Rate (%)', data=covid_period, label='During COVID-19')
plt.title('Unemployment Rate during COVID-19 Period')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()


# In[24]:


# Calculate summary statistics
print("\nDataset 1 - Unemployment Rate Summary:")
print(df1[' Estimated Unemployment Rate (%)'].describe())

print("\nDataset 2 - Unemployment Rate Summary:")
print(df2[' Estimated Unemployment Rate (%)'].describe())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




