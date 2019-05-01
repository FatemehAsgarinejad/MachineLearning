#Random Forest
"""
Ensemble learning is when we take a single algorithm multiple times or 
different algorithms in order to have a more accurate model and more powerful
than the original.
Like implementing decision tree and gaining Random Forest

Step1: Pick at random k data points from the training set
Step2: Build the decision tree associated to these k data points
Step3: choose the number of trees we want to buld and repeat Steps 1 and 2
Step4: For a new data point make each of the trees predict for it and then assign
the data to the average accross all the predicted values.
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("/Users/fatemeh/Desktop/HW_Topic7/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting the SVR to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 300, random_state=0)
regressor.fit(X, Y)

#Predicting a new result
y_pred= regressor.predict(6.5)

#But the Decision Tree Regression should not be continuous, instead step by step
#Smoother curve plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color='navy')
plt.plot(X_grid, regressor.predict(X_grid), color='gray')
plt.title("Truth or Bluff(Random Forest Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



