#R-squared | Logistic Regression
"""
SSres = SUM(yi - yi^)^2 (observations to the fitted line)
SStot = SUM(yi - yavg)^2 (observations to the average)
R^2 = 1 - SSres/SStot
The closer the R^2 to 1, the better
The R^2 determines how well the model is set to the data
R^2 is the goodness of fitness
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Setting working directory
#Importing dataset
address = '/Users/fatemeh/Desktop/P14-Logistic-Regression/Logistic_Regression'
dataset = pd.read_csv(address + "/Social_Network_Ads.csv")
#print(data.head())
#distinguish the features and dependent columns 
X = dataset.iloc[:,2:-1].values
Y = dataset.iloc[:,-1].values
#Encoding categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder() #Creating an object from LabelEncoder class
X[:,0] =labelencoder_X.fit_transform(X[:,0])

#dividing the dataset into two parts, train set and test set
from sklearn.model_selection import train_test_split
#building train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Evaluating the performance of logistic regression using confusion matrix
from sklearn.metrics import confusion_matrix #Function
cm = confusion_matrix(Y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1 , stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1 , stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
#By colorising all the pixel points
from matplotlib.colors import ListedColormap #Class ListedColormap helps colorizing all the points
X_set, y_set = X_test, y_test #creating some local variables
#figuring out the range of the axis
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#coloring the pixels, based on what classifier predicts
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.8, cmap = ListedColormap(('pink', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#plotting the points in the plot
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'navy'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
