#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error
data=pd.read_csv('heart.csv')


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


from sklearn.preprocessing  import LabelEncoder


# In[11]:


le=LabelEncoder()


# In[12]:


data.Age=le.fit_transform(data.Age)
data.Sex=le.fit_transform(data.Sex)
data.ChestPainType=le.fit_transform(data.ChestPainType)
data.RestingBP=le.fit_transform(data.RestingBP)
data.Cholesterol=le.fit_transform(data.Cholesterol)
data.FastingBS=le.fit_transform(data.FastingBS)
data.RestingECG=le.fit_transform(data.RestingECG)
data.MaxHR=le.fit_transform(data.MaxHR)
data.ExerciseAngina=le.fit_transform(data.ExerciseAngina)
data.Oldpeak=le.fit_transform(data.Oldpeak)
data.ST_Slope=le.fit_transform(data.ST_Slope)


# In[13]:


data.HeartDisease=le.fit_transform(data.HeartDisease)


# In[14]:


data.head()


# In[16]:


x=data.iloc[ : ,:11]
y=data['HeartDisease']


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=2)


# In[18]:


reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print('Coefficients: ', reg.coef_)
rmse = math.sqrt(mean_squared_error(y_test,y_pred))
rmse


# In[19]:


plt.scatter(x_test.Cholesterol, y_pred, label = 'Predicted')
plt.scatter(x_test.Cholesterol, y_test, label = 'Actual')
plt.legend()


# In[ ]:




