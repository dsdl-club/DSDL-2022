#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.axes as ax


# # Little Bit Data Exploration

# In[2]:


data=pd.read_csv('heart.csv')


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


data.describe(include='all')


# In[9]:


data_dup=data.duplicated().any()


# In[10]:


data_dup


# # Data Visualizition

# In[11]:


sns.pairplot(data)


# # Data Processing

# In[12]:


data['Sex'].unique()


# In[13]:


data['ExerciseAngina'].unique()


# In[14]:


data['Sex']=data['Sex'].map({'M':0,'F':1})


# In[15]:


data['ExerciseAngina']=data['ExerciseAngina'].map({'N':0,'Y':1})


# In[16]:


data


# In[17]:


catg=[]
nume=[]

for column in data.columns:
    if data[column].nunique() <=10:
        catg.append(column)
        
    else:
        nume.append(column)


# In[18]:


catg


# In[19]:


nume


# # Convert Columns From String To Numerical Values

# Encoding Categorical Data

# In[20]:


catg


# In[21]:


data['ChestPainType'].unique()


# In[22]:


catg.remove('Sex')
catg.remove('ExerciseAngina')
catg.remove('HeartDisease')
data=pd.get_dummies(data,columns=catg,drop_first=True)


# In[23]:


data.head()


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


st=StandardScaler()
data[nume]=st.fit_transform(data[nume])


# In[26]:


data.head()


# # Splitting the Dataset Into The Training Set And Test Set

# In[27]:


X=data.drop('HeartDisease',axis=1)


# In[28]:


Y=data['HeartDisease']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[31]:


X_train


# In[32]:


X_test


# # LogisticRegression

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


log=LogisticRegression()
log.fit(X_train,Y_train)


# In[35]:


y_pred1=log.predict(X_test)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


accuracy_score(Y_test,y_pred1)


# # SVC

# In[38]:


from sklearn import svm


# In[39]:


svm =svm.SVC()


# In[40]:


svm.fit(X_train,Y_train)


# In[41]:


y_pred2=svm.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy_score(Y_test,y_pred2)


# # KNeighbors Classifier

# In[44]:


from sklearn.neighbors import KNeighborsClassifier


# In[45]:


knn=KNeighborsClassifier(n_neighbors=2)


# In[46]:


knn.fit(X_train,Y_train)


# In[47]:


y_pred3=knn.predict(X_test)


# In[48]:


from sklearn.metrics import accuracy_score


# In[49]:


accuracy_score(Y_test,y_pred3)


# # Non-Linear Algorithms

# In[50]:


data=pd.read_csv('heart.csv')


# In[51]:


data.head()


# In[52]:


data['Sex']=data['Sex'].map({'M':0,'F':1})


# In[53]:


data['ChestPainType']=data['ChestPainType'].map({'ATA':0,'NAP':1,'ASY':2,'TA':3})


# In[54]:


data['RestingECG']=data['RestingECG'].map({'Normal':0,'ST':1,'LVH':2})


# In[55]:


data['ExerciseAngina']=data['ExerciseAngina'].map({'N':0,'Y':1})


# In[56]:


data['ST_Slope']=data['ST_Slope'].map({'Up':0,'Flat':1,'Down':2})


# In[57]:


X=data.drop('HeartDisease',axis=1)
Y=data['HeartDisease']


# In[58]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=45)


# # Decision Tree Classifier

# In[59]:


from sklearn.tree import DecisionTreeClassifier


# In[60]:


dt=DecisionTreeClassifier()


# In[61]:


dt.fit(X_train,Y_train)


# In[62]:


y_pred4=dt.predict(X_test)


# In[63]:


from sklearn.metrics import accuracy_score 


# In[64]:


accuracy_score(Y_test,y_pred4)


# # Random Forest Calssifier

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


rf=RandomForestClassifier()


# In[67]:


rf.fit(X_train,Y_train)


# In[68]:


y_pred5=rf.predict(X_test)


# In[69]:


from sklearn.metrics import accuracy_score


# In[70]:


accuracy_score(Y_test,y_pred5)


# # Gradient Boosting Classifier

# In[71]:


from sklearn.ensemble import GradientBoostingClassifier


# In[72]:


gb=GradientBoostingClassifier()


# In[73]:


gb.fit(X_train,Y_train)


# In[74]:


y_pred6=gb.predict(X_test)


# In[75]:


from sklearn.metrics import accuracy_score


# In[76]:


accuracy_score(Y_test,y_pred6)


# In[77]:


final_data=pd.DataFrame({'Model':['LR','SVM','KNN','DT','RF','GB'],
                         'ACC':[accuracy_score(Y_test,y_pred1),
                                accuracy_score(Y_test,y_pred2),
                                accuracy_score(Y_test,y_pred3),
                                accuracy_score(Y_test,y_pred4),
                                accuracy_score(Y_test,y_pred5),
                                accuracy_score(Y_test,y_pred6)]})


# In[78]:


final_data


# In[79]:


sns.barplot(final_data['Model'],final_data['ACC'])


# In[80]:


data.head()


# In[81]:


X=data.drop('HeartDisease',axis=1)
Y=data['HeartDisease']


# In[82]:


X


# In[83]:


from sklearn.ensemble import GradientBoostingClassifier


# In[84]:


gb=GradientBoostingClassifier()
gb.fit(X,Y)


# In[85]:


data.head()


# # Prediction on New Data

# In[86]:


import pandas as pd


# In[87]:


new_data=pd.DataFrame({
    'Age':59,
    'Sex':1,
    'ChestPainType':1,
    'RestingBP':130,
    'Cholesterol':289,
    'FastingBS':0,
    'RestingECG':0,
    'MaxHR':98,
    'ExerciseAngina':0,
    'Oldpeak':1.5,
    'ST_Slope':0,
},index=[0])     
    


# In[88]:


new_data


# In[89]:


h=rf.predict(new_data)
if h[0]==0:
    print('No Disease')
else:
    print('Disease')    


# # Saving Model Using Joblib

# In[90]:


import joblib


# In[91]:


joblib.dump(rf,'model_joblib_heart')


# In[92]:


model=joblib.load('model_joblib_heart')


# In[93]:


model.predict(new_data)


# # GUI

# In[94]:


from tkinter import *
import joblib


# In[96]:


def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    p7=float(e7.get())
    p8=float(e8.get())
    p9=float(e9.get())
    p10=float(e10.get())
    p11=float(e11.get())
   
    model=joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]])
    
    if result==0:
        Label(master,text='No Heart Disease').grid(row=30)
        
    else:
        Label(master,text='Possibility of Heart Diseases').grid(row=31)
        
master= Tk()
master.title("UttkarshSahai(2125ce1035)")

label=Label(master,text="Heart Disease Prediction System",
            bg='white',fg='green').grid(row=0,columnspan=2)

Label(master,text="Enter Your Age").grid(row=1)
Label(master,text="Male or Female {'M':0,'F':1}").grid(row=2)
Label(master,text="ChestPainType {'ATA':0,'NAP':1,'ASY':2,'TA':3}").grid(row=3)
Label(master,text="Enter Value of RestingBP").grid(row=4)
Label(master,text="Enter  value of Cholesterol").grid(row=5)
Label(master,text="Enter vakue of FastingBS").grid(row=6)
Label(master,text="RestingECG {'Normal':0,'ST':1,'LVH':2}").grid(row=7)
Label(master,text="Enter Value of MaxHR").grid(row=8)
Label(master,text="ExerciseAngina {'N':0,'Y':1}").grid(row=9)
Label(master,text="Enter value of Oldpeak").grid(row=10)
Label(master,text="ST_Slope {'Up':0,'Flat':1,'Down':2}").grid(row=11)

                         
e1=Entry(master)
e2=Entry(master)       
e3=Entry(master)    
e4=Entry(master)    
e5=Entry(master)    
e6=Entry(master)    
e7=Entry(master) 
e8=Entry(master)
e9=Entry(master)
e10=Entry(master)
e11=Entry(master)                                               

                         
e1.grid(row=1,column=1) 
e2.grid(row=2,column=1)  
e3.grid(row=3,column=1)  
e4.grid(row=4,column=1)  
e5.grid(row=5,column=1)  
e6.grid(row=6,column=1)  
e7.grid(row=7,column=1) 
e8.grid(row=8,column=1) 
e9.grid(row=9,column=1)
e10.grid(row=10,column=1) 
e11.grid(row=11,column=1) 
                                               
Button(master,text="Predict",command=show_entry_fields).grid(row=12)

mainloop()  


# In[ ]:




