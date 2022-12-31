import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error
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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=2)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
rmse = math.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)