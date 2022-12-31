#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


data=pd.read_csv(r"C:\Users\akshi\Downloads\DSDL\heart.csv")


# In[31]:


data.head()


# In[32]:


data.info()


# In[33]:


x=data[['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]


# In[34]:


y=data['HeartDisease']


# In[59]:


x=pd.get_dummies(data,columns=(['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']))


# In[60]:


from sklearn.preprocessing  import LabelEncoder


# In[61]:


le=LabelEncoder()


# In[62]:


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


# In[63]:


data.HeartDisease=le.fit_transform(data.HeartDisease)


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=1)


# In[66]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# In[67]:


y_pred = classifier.predict(x_test)


# In[68]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[69]:


print(cm)


# In[70]:


print(ac)


# In[ ]:




