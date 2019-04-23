#Simple Linear Regression
#Importing the dataset
dataset = read.csv("/Users/fatemeh/Desktop/P14-Simple-Linear-Regression/Simple_Linear_Regression/Salary_Data.csv")
#Splitting the dataset to training and test set
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
#Fitting the simple linear regression to training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#Visualising the training set results
#install.packages("ggplot2",dependencies=TRUE)
#library(ggplot2)
#Step by step :
#plotting all the observation points
#plotting the regression line
#add tile and labels
#Visualising the training set
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'tomato') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            color = 'navy') +
  ggtitle('Salary vs Experience(Training set)') + 
  xlab('Years of Experience') +
  ylab('Salary')
#Visualising the Test set
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             color= "green") +
  geom_line(aes(x = test_set$YearsExperience, y = y_pred),
            color = 'gold') +
  ggtitle('Salary vs Experience(Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#Fitting the regressor on the training data
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#predicting a single value
predict(regressor, data.frame(YearsExperience = 1.5)) 