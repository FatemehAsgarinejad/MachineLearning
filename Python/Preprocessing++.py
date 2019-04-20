#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:07:13 2019

@author: fatemeh
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Setting working directory
#Importing dataset
dataset = pd.read_csv("/Users/fatemeh/Desktop/+ Machine Learning/Part 1 - Data Preprocessing/Data.csv")
#print(data.head())
#distinguish the features and dependent columns 
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
#Taking care of missing data
#Substituiting the missing values with mean of the columns, mode, etc are the possible approaches
from sklearn.preprocessing import Imputer #class imputer 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #creating an object of the class imputer
imputer = imputer.fit(X[:,1:3]) #fitting the imputer with the dataset
X[:,1:3] = imputer.transform(X[:,1:3])
#Pressing command+I would pop up some information about each function, class, etc

#Encoding categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder() #Creating an object from LabelEncoder class
X[:,0] =labelencoder_X.fit_transform(X[:,0])

#Encoding the categorial values for Y column
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

##########################################################################
#2
#We use dummy variables to create one column for each country, valued 1 if is the country and values 0 otherwise
#Then we make the number of columns as much as the number of categories.
from sklearn.preprocessing import OneHotEncoder #a class for declaring dummy variables
onehotencoder = OneHotEncoder(categorical_features=[0]) #creating an object from the class OneHotEncoder
X = onehotencoder.fit_transform(X).toarray()

##########################################################################
#3
#dividing the dataset into two parts, train set and test set
from sklearn.model_selection import train_test_split
 #building train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
##########################################################################
#4
#feature scaling: scaling the values and normalizing them.
#Standardisation: Xstand = x - mean(x) / standard deviation(x)
#Normalisation: Xnorm = x - min(x) / max(x) - min(x)

#Importing standard scaler class for feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #creating an object of the class 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#Because Y column is a categorical division, there is no need to apply feature selection to it
#But in regression, feature selection should be applied to Y column as well.




















