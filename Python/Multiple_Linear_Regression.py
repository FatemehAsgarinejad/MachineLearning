#Multiple Linear Regression

#Importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset= pd.read_csv("/Users/fatemeh/Desktop/P14-Multiple-Linear-Regression/Multiple_Linear_Regression/50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Encoding the independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variables trap
X = X[:, 1:] #removing 1st column of X

#Splitting the dataset to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#Adding a column for b0
X = np.append(arr =np.ones((50, 1)).astype(int),values = X, axis =1)

#Creating Optimal matrix of features, called x_optimal
X_opt = X[:,[0,1,2,3,4,5]]
#selecting a significance level 
sl = 0.05
#fit the model with all possible predictors
regressor_OLS = sm.OLS(endog= Y, exog = X_opt).fit()
regressor_OLS.summary()
#removing the independent value with highest p-value

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog= Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog= Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog= Y, exog = X_opt).fit()
regressor_OLS.summary()
