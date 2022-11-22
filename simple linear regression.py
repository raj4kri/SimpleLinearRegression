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

