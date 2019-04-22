#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:15:07 2019

@author: fatemeh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("/Users/fatemeh/Desktop/+ Machine Learning/Part 1 - Data Preprocessing/Data.csv")
dataset.head()

#preprocessing
"""
- dividing independent and dependent values
- missing values : imputer
- categorical to numeric : labelencoder and onehotencoder
- splitting test and training set train_test_split from model_selection
- scaling with standard scaler
"""
#Dividing the dataset into independent and dependent values
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Filling the missing data with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])

#Converting categorical data to numeric
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
labelencoder_X = LabelEncoder() #Creating an object from LabelEncoder class
X[:,0] =labelencoder_X.fit_transform(X[:,0])

#Adding new columns to x for the first column
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#splitting the dataset to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#scaling the data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)




