#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import seaborn as sns


# In[2]:


data=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')


# A Little Bit Data Exploration

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.dtypes


# In[9]:


data.shape


# In[10]:


x=data.drop(data.columns[6],axis=1)  
y=data['seller_type']


# In[11]:


x.head()


# In[12]:


y.head()


# In[13]:


y.unique()


# In[14]:


y.value_counts()


# In[15]:


x=pd.get_dummies(data,columns=(['fuel','name','seller_type','transmission','owner']))


# In[16]:


x.head()


# In[17]:


y=pd.get_dummies(data,columns=(['seller_type']))


# In[18]:


y.head()


# In[19]:


sns.pairplot(data)


# In[20]:


x.shape


# In[21]:


y.shape


# In[22]:


data['seller_type'].unique()


# In[23]:


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y['seller_type_Individual'],test_size=0.30,random_state=0)


# Feature Scaling

# In[24]:


from sklearn.preprocessing import StandardScaler

le=StandardScaler()
x_train=le.fit_transform(xtrain)
xtest=le.transform(xtest)


# In[25]:


xtrain.head()


# In[26]:


from sklearn.linear_model import LinearRegression

linear_regressor=LinearRegression()
linear_regressor.fit(xtrain,ytrain)


# In[27]:


predicted_value=linear_regressor.predict(xtrain)


# In[28]:


predicted_value


# In[29]:


ytrain


# In[30]:


from sklearn.metrics import mean_squared_error

error=mean_squared_error(ytrain,predicted_value)


# In[31]:


error


# In[32]:


# original hypothesis
plt.plot(xtrain,ytrain,"*",color="green")

# model hypothesis
plt.plot(ytrain,predicted_value,"+",color="red")

plt.xlabel('input')
plt.ylabel('output')
plt.show

