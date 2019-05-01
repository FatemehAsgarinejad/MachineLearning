#Decision tree regression model
"""
CART: classification and regression trees
Not good for a single dimentional
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
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, Y)

#Predicting a new result
y_pred= regressor.predict(6.5)

#Visualising the SVR results
plt.scatter(X, Y, color='navy')
plt.plot(X, regressor.predict(X), color='gray')
plt.title("Truth or Bluff(Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#But the Decision Tree Regression should not be continuous, instead step by step
#Smoother curve plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color='navy')
plt.plot(X_grid, regressor.predict(X_grid), color='gray')
plt.title("Truth or Bluff(Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



