#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Salary_Data dataset
"""
Regression models (linear and non-linear) are used for predicting a real value.
Regression Techniques vary from linear Regression to SVR and Random Forest Regression.
Simple Linear Regression
Multiple Linear Regression
Polynomial Regression
Support Vector for Regression(SVR)
Decision Tree Classification
Random Forest Classification
"""
#SLR: Simple Linear Regression Y = b0 + b1*x1
#How to find the best line fitting data? yi where the point is, yi^ where it should be on the line,
#Sum of (yi - yi^)^2 for all the points gives a value which we calculate for all lines and would choose the minimum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("/Users/fatemeh/Desktop/P14-Simple-Linear-Regression/Simple_Linear_Regression/Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values #Y is a vector of dependent variables

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state =0)

#Simple Linear Regression Method would definitely take care of the scaling
#Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression #Importing the class linearRegression
linearRegressor = LinearRegression() #Creating an object from the class
linearRegressor.fit(X_train, y_train) #fitting the regressor to the training data


#Predicting the results for the test set.
y_pred = linearRegressor.predict(X_test) #Creating a vector of the predicted values

#Visualising the training set results
plt.scatter(X_train, Y_train, color="gold")
plt.plot(X_train, linearRegressor.predict(X_train), color="tomato") #simple linear regression line
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

#Visualising the test set results
plt.scatter(X_test, Y_test, color="gray")
plt.plot(X_train, linearRegressor.predict(X_train), color="navy")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

#predicting a single value
regressor.predict(5)






