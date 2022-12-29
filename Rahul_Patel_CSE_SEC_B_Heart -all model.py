#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data =pd.read_csv(r'C:\Users\smara\Downloads\heart data set.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


y=data['HeartDisease']
x=data[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]


# In[9]:


x.shape


# In[10]:


y.shape


# In[11]:


data.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


le=LabelEncoder()


# In[14]:


data.Sex=le.fit_transform(data.Sex)
data.ChestPainType=le.fit_transform(data.ChestPainType)
data.RestingBP=le.fit_transform(data.RestingBP)
data.RestingECG=le.fit_transform(data.RestingECG)
data.ExerciseAngina=le.fit_transform(data.ExerciseAngina)
data.ST_Slope=le.fit_transform(data.ST_Slope)
data.Age=le.fit_transform(data.Age)
data.Cholesterol=le.fit_transform(data.Cholesterol)
data.FastingBS=le.fit_transform(data.FastingBS)
data.MaxHR=le.fit_transform(data.MaxHR)
data.Oldpeak=le.fit_transform(data.Oldpeak)
data.HeartDisease=le.fit_transform(data.HeartDisease)


# In[15]:


data.head()


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model=LinearRegression()


# In[18]:


x=pd.get_dummies(data,columns=(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']))


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)


# In[21]:


x_train


# In[22]:


y_train


# In[23]:


x_test


# In[24]:


y_test


# In[25]:


#Linear Regression


# In[26]:


model.fit(x_test,y_test)


# In[27]:


model.intercept_


# In[28]:


model.coef_


# In[29]:


y_pred=model.predict(x)


# In[30]:


from sklearn.metrics import mean_absolute_error


# In[31]:


mean_absolute_error(y,y_pred)


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


y_pred=model.predict(x)


# In[34]:


y_test


# In[35]:


model.score(x_test,y_test)


# # Prediction

# In[36]:


x_test.iloc[-1,:]


# In[37]:


model.predict([x_test.iloc[-1,:]])


# In[38]:


y_test.iloc[-1:]


# # Logistic Regression
# 

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logistic=LogisticRegression(max_iter=500)


# In[41]:


logistic.fit(x_test,y_test)


# In[42]:


y_pred=logistic.predict(x)


# In[43]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[44]:


accuracy_score(y,y_pred)


# In[45]:


confusion_matrix(y,y_pred)


# In[46]:


print(classification_report(y,y_pred))


# # Prediction

# In[47]:


x_test.iloc[-1,:]


# In[48]:


logistic.predict([x_test.iloc[-1,:]])


# In[49]:


y_test.iloc[-1:]


# # Decision Tree

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


df=DecisionTreeClassifier()


# In[52]:


df.fit(x,y)


# In[53]:


y_pred=df.predict(x)


# In[54]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[55]:


accuracy_score(y,y_pred)


# In[56]:


confusion_matrix(y,y_pred)


# In[57]:


print(classification_report(y,y_pred))


# # Prediction

# In[58]:


x_test.iloc[-1,:]


# In[59]:


df.predict([x_test.iloc[-1,:]])


# In[60]:


y_test.iloc[-1:]


# # Random Forest

# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf=RandomForestClassifier()


# In[63]:


rf.fit(x,y)


# In[64]:


y_pred=df.predict(x)


# In[65]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[66]:


accuracy_score(y,y_pred)


# In[67]:


confusion_matrix(y,y_pred)


# # Prediction

# In[68]:


x_test.iloc[-1,:]


# In[69]:


rf.predict([x_test.iloc[-1,:]])


# In[70]:


y_test.iloc[-1:]


# # KNN

# In[71]:



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[72]:


knn.fit(x,y)


# In[73]:


y_pred=knn.predict(x)


# In[74]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[75]:


accuracy_score(y,y_pred)


# In[76]:


confusion_matrix(y,y_pred)


# In[77]:


print(classification_report(y,y_pred))


# # Prediction

# In[78]:


x_test.iloc[-1,:]


# In[79]:


knn.predict([x_test.iloc[-1,:]])


# In[80]:


y_test.iloc[-1:]


# In[ ]:




