# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:18:55 2020

@author: ASUS
"""

import os

import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree, svm ,linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
from sklearn.metrics import mean_squared_error


os.chdir("E:\edwisor data science")

df = pd.read_csv("day.csv")

df.info()
df.describe()


## converting few variables in from numberic to category

df['season'] = df['season'].astype('category')
df['yr'] = df['yr'].astype('category')
df['mnth'] = df['mnth'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['weathersit'] = df['weathersit'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['holiday'] = df['holiday'].astype('category')




#removing unnecessary columns



df.drop_duplicates(keep= 'first', inplace= True)

df.isnull().sum()

df['workingday'].nunique()

df['casual'].describe()


#############normal distribution of cnt variable########################

plt.hist(df['cnt'], bins = 'auto')




# detetcing outliers 

cols = ['temp','atemp','hum', 'windspeed','cnt']


fig, axes = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(25,15)

#-- Plot total counts on y bar
sn.boxplot(data=df, y="cnt",ax=axes[0][0])

#-- Plot temp on y bar
sn.boxplot(data=df, y="temp",ax=axes[0][1])

#-- Plot atemp on y bar
sn.boxplot(data=df, y="atemp",ax=axes[0][2])

#-- Plot hum on y bar
sn.boxplot(data=df, y="hum",ax=axes[0][3])

#-- Plot windspeed on y bar
sn.boxplot(data=df, y="windspeed",ax=axes[1][0])

#-- Plot total counts on y-bar and 'yr' on x-bar
sn.boxplot(data=df,y="cnt",x="yr",ax=axes[1][1])

#-- Plot total counts on y-bar and 'mnth' on x-bar
sn.boxplot(data=df,y="cnt",x="mnth",ax=axes[1][2])

#-- Plot total counts on y-bar and 'date' on x-bar
sn.boxplot(data=df,y="cnt",x="dteday",ax=axes[1][3])

#-- Plot total counts on y-bar and 'season' on x-bar
sn.boxplot(data=df,y="cnt",x="season",ax=axes[2][0])

#-- Plot total counts on y-bar and 'weekday' on x-bar
sn.boxplot(data=df,y="cnt",x="weekday",ax=axes[2][1])

#-- Plot total counts on y-bar and 'workingday' on x-bar
sn.boxplot(data=df,y="cnt",x="workingday",ax=axes[2][2])

#-- Plot total counts on y-bar and 'weathersit' on x-bar
sn.boxplot(data=df,y="cnt",x="weathersit",ax=axes[2][3])






for i in cols:
    q75, q25 = np.percentile(df.loc[:,i], [75,25])
    iqr = q75 - q25
    
    minimum  = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    df.loc[df.loc[:,i] < minimum, i] = np.nan
    df.loc[df.loc[:,i] > maximum, i] = np.nan


## checking how many outliers are there by imputing null values

df.isnull().sum()
## very less outliers, so keeping them as it is 


## correlation 

corr = df.corr()



corr_matrix = df.corr()
plt.subplots(figsize=(12,9))
sn.heatmap(corr_matrix, vmax=0.9, square=True)



##################graph of individual categorical features by count###################
fig, saxis = plt.subplots(3, 3,figsize=(16,12))

sn.barplot(x = 'season', y = 'cnt',hue= 'yr', data=df, ax = saxis[0,0], palette ="Blues_d")
sn.barplot(x = 'yr', y = 'cnt', order=[0,1,2,3], data=df, ax = saxis[0,1], palette ="Blues_d")
sn.barplot(x = 'mnth', y = 'cnt', data=df, ax = saxis[0,2])
sn.barplot(x = 'holiday', y = 'cnt' , data=df, ax = saxis[1,0])
sn.barplot(x = 'weekday', y = 'cnt',  data=df, ax = saxis[1,1])
sn.barplot(x = 'workingday', y = 'cnt', data=df, ax = saxis[1,2])
sn.barplot(x = 'weathersit', y = 'cnt',order=[1,2,3,4] ,data=df, ax = saxis[2,0])
sn.barplot(x = 'dteday', y = 'cnt' , data=df, ax = saxis[2,1])

#sn.pointplot(x = 'weathersit', y = 'cnt', data=bike, ax = saxis[2,0])





## dropping atemp variable as it is highly correalted with temp
## dropping "casual" and "registered" variables as their addition is cnt. 
## Treating cnt as target variable
## holidayand workingday seem to explain same thing...dropping holiday
## instant and dteday don't seem to be explaining much. so dropping them




col = ['instant', 'dteday','holiday','casual','registered', 'atemp']

df = df.drop(columns = col)



##############dummies of the categorical variables###############################

df= pd.get_dummies(df, drop_first = True)

## rearranging the variables

df1 = df.iloc[:,0:4]
df2 = df.iloc[:,4:28]


df_main = pd.concat([df2,df1], axis=1)



         
x = df_main.iloc[:,0:27]
y = df_main.iloc[:,27]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


      
 ################################ linear regression##########################


model = sm.OLS(y_train, x_train).fit()    

model.summary()


predict_LR = model.predict(x_test)



## error metrics

def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true -y_pred) / y_true))
    return mape

error_LR = MAPE(y_test, predict_LR)



mean_squared_error(y_test, predict_LR)

np.sqrt(mean_squared_error(y_test, predict_LR))




########################## random forest########################################

rf = RandomForestRegressor(n_estimators = 500).fit(x_train, y_train);

rf_predict = rf.predict(x_test)


## error metrics
mean_squared_error(y_test, rf_predict)

np.sqrt(mean_squared_error(y_test, rf_predict))

errors_RF = MAPE(y_test, rf_predict)
r_squared = model.score(x_test, y_test)


ref = pd.DataFrame(rf_predict)


ref = ref.rename(columns= {"index":"columns" , 0 : "predicted"})

plt.hist(ref['predicted'], bins = 'auto')

actual = pd.DataFrame(y_test)
actual = actual.rename(columns= {"index":"columns" , 0 : "real"})


plt.hist(actual['real'])


#########################CROSS VALIDATION#########################################




logreg = LinearRegression()

scores_LR = cross_val_score(logreg, x, y, cv=5, scoring='mean_squared_error')


mse_scores = -scores_LR
print(mse_scores)
print(mse_scores.mean())
print (np.sqrt(mse_scores.mean()))





































