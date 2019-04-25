"""
Regression Template
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("/Users/fatemeh/Desktop/Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

#Splitting the dataset into training set and test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""
#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
X_train = sc_Y.fit_transform(Y_train)
"""

#Importing the regression model to dataset

#Fitting the regression model to the dataset

#predicting a new result with polynomial regression
y_pred = regressor.predict(6.5)

#Visualising the Polinomial Linear Regression Results
plt.scatter(X, Y, color="tomato")
plt.plot(X, regressor.predict(X), color="navy") #xpoly or poly_reg.fit_transform(X)
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising the Polinomial Linear Regression Results (high resolution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_grid, Y, color="tomato")
plt.plot(X_grid, regressor.predict(X), color="navy") #xpoly or poly_reg.fit_transform(X)
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()