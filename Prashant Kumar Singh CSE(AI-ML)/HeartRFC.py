import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score
data=pd.read_csv('Flask_ML\heart.csv')
gender = {'M': 1,'F': 2}
Type = {'ATA': 1,'NAP': 2, 'ASY': 3, 'TA': 4}
ECG = {'Normal': 1,'ST': 2, 'LVH': 3}
ST = {'Up': 1,'Flat': 2, 'Down': 3}
eni = {'N': 1,'Y': 2}
data.Sex = [gender[item] for item in data.Sex]
data.ChestPainType = [Type[item] for item in data.ChestPainType]
data.RestingECG = [ECG[item] for item in data.RestingECG]
data.ST_Slope = [ST[item] for item in data.ST_Slope]
data.ExerciseAngina = [eni[item] for item in data.ExerciseAngina]
x=data.iloc[ : ,:11]
y=data['HeartDisease']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=3)
'''a_score=[]
for i in  range(1, 100):
    clf = RandomForestClassifier(n_estimators = i) 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    a_score.append(accuracy_score(y_test, y_pred))
    
max(a_score)'''
'''a_score.index(max(a_score))'''
clf = RandomForestClassifier(n_estimators = 33) 
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
HRFCacc=accuracy_score(y_test, y_pred)