#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:46:34 2019

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
#We use dummy variables to create one column for each country, valued 1 if is the country and values 0 otherwise


























from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])














