#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r"C:\Users\akshi\Downloads\DSDL\heart.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


x=data[['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]


# In[6]:


y=data['HeartDisease']


# In[7]:


x=pd.get_dummies(data,columns=(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']))


# In[22]:


x.head()


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


from sklearn.preprocessing  import LabelEncoder


# In[10]:


le=LabelEncoder()


# In[11]:


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


# In[12]:


data.HeartDisease=le.fit_transform(data.HeartDisease)


# In[13]:


data.head()


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=1)


# In[16]:


lr=LogisticRegression()


# In[17]:


lr.fit(x_train,y_train)


# In[18]:


y_pred = lr.predict(x_test)


# In[19]:


print(lr.coef_)
print(lr.intercept_)


# In[20]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

