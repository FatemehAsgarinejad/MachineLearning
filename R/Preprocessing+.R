#Importing libraries: R already contains the reuired libraries in packages tab
#1
#Data Preprocessing 
#Importing the dataset
dataset = read.csv("/Users/fatemeh/Desktop/+ Machine Learning/Part 1 - Data Preprocessing/Data.csv")
#Unlike Python, indexes start at 1 not 0

#Taking care of missing value in R
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Age, FUN=function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)
#################################################
#2
#Encoding categorical data (Country column)
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))  #c is a vector in R

#Encoding categorical data (Purchased column)
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes', 'No'),
                           labels = c(1,0))




