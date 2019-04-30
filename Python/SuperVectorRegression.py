"""SVR: Support Vector Regression"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("/Users/fatemeh/Desktop/HW_Topic7/Position_Salaries.csv")
#distinguish the features and dependent columns 
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Feature Scaling is needed in SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
#Since scaler requires a 2D value, an error would pop out
Y = np.squeeze(sc_Y.fit_transform(Y.reshape(-1, 1)))


#Fitting the SVR to the dataset
from sklearn.svm import SVR #Importing the class SVR
#Kernel determines wether we want a linear, polynomial or a gaussian SVR
regressor = SVR(kernel = 'rbf') #Creating an object from the class SVR
regressor.fit(X, Y)

#Predicting a new result
y_pred= sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising the SVR results
plt.scatter(X, Y, color='navy')
plt.plot(X, regressor.predict(X), color='gray')
plt.show()
















