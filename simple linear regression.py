_# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:19:20 2022

@author: Raj Aryan
"""

#import the models
import pandas as pd

#read csv file

dataset = pd.read_csv("R:\simple linear regresstion/01students.csv")
df = dataset.copy()

#split the dataset vertically into x and y

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

#split the dataset by rows into training and test datasets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1234)


#create and train the simple linear regression

from sklearn.linear_model import LinearRegression

#regressor

std_reg = LinearRegression()

#train or fit the trainning data

std_reg.fit(x_train,y_train)

#now ready to do predict the value of y from test data

y_pred = std_reg.predict(x_test)

#calculate the r-sqared and equation of line

slr_score = std_reg.score(x_test,y_test)

#coeffient of the line
slr_coeff = std_reg.coef_ 
slr_intercept = std_reg.intercept_ 
 

#RMSE root mean squared error

from sklearn.metrics import mean_squared_error
import math

slr_rsme = math.sqrt(mean_squared_error(y_test,y_pred))



#plotting the result using matplotlib
import  matplotlib.pyplot as plt

plt.scatter(x_test,y_test)

#trentline

plt.plot(x_test, y_pred)
plt.ylim(ymin=0)
plt.show()







