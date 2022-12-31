#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score
data=pd.read_csv('heart.csv')


# In[2]:


from sklearn.preprocessing  import LabelEncoder


# In[3]:


le=LabelEncoder()


# In[4]:


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


# In[5]:


data.HeartDisease=le.fit_transform(data.HeartDisease)


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


x=data.iloc[ : ,:11]
y=data['HeartDisease']


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=2)


# In[10]:


clf = RandomForestClassifier(n_estimators = 33) 
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)


# In[11]:


plt.scatter(x_test.Cholesterol, y_pred, label = 'Predicted')
plt.scatter(x_test.Cholesterol, y_test, label = 'Actual')
plt.legend()


# In[ ]:




