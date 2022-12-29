#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv(r'C:\Users\smara\Downloads\heart data set.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.columns


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


x=data[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
y=data['HeartDisease']


# In[9]:


x=pd.get_dummies(data,columns=(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']))


# In[10]:


x.head()


# In[11]:


from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()


# In[12]:


data.Age=le.fit_transform(data.Age)


# In[13]:


data.Sex=le.fit_transform(data.Sex)


# In[14]:


data.ChestPainType=le.fit_transform(data.ChestPainType)


# In[15]:


data.RestingBP=le.fit_transform(data.RestingBP)


# In[16]:


data.Cholesterol=le.fit_transform(data.Cholesterol)
data.FastingBS=le.fit_transform(data.FastingBS)
data.RestingECG=le.fit_transform(data.RestingECG)
data.MaxHR=le.fit_transform(data.MaxHR)
data.ExerciseAngina=le.fit_transform(data.ExerciseAngina)
data.Oldpeak=le.fit_transform(data.Oldpeak)
data.ST_Slope=le.fit_transform(data.ST_Slope)


# In[17]:


data.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7,random_state=51)


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


regressior=DecisionTreeClassifier(criterion='gini')


# In[22]:


regressior.fit(x_train,y_train)


# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


regressior.score(x_test,y_test)


# In[25]:



from sklearn.metrics import mean_squared_error


# In[ ]:





# In[ ]:




