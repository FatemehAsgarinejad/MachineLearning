#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple linear regression
we want to understand which independent factors are moe important.
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
Dummy variables
When we have a column with n categories, we add up n-1 columns as dummy varibales, because from them 
it is easy to understand the third column.
p-value: 
    take a sample, calculate its mean, call it x-bar
    p(mean(total) >= x-bar |  H0 True) more than significane level (0.05 for instance) then H0 is True
    if p-value is high, Null is right
    if p-value is low, Null must go

Building a model  (Step by Step)
- We get rid of some unusefull columns
5 methods of building models
1- All in: put all variables in
2- Backward elimination: sl(significance level) = 0.05 ,
   if predictor have a p-value above sl, remove predictor, other wise finish.
3- Forward selection : sl = 0.05,
   fit all predictors, choose the one with the lowest p-value, keep it and fit
   all the possible models with one extra predictor added to the ones you already have.
   consider the predictor with the lowest pvalue if it is less than sl, add it, otherwise finish.
4- Bidirectional Elimination: forward then backward and forward again
5- Score comparison: select a criterian of goodness, construct all possible regression models,
   with 2^n -1 total combinations and select the one with the best criterion.
2,3 and 4 are considered as stepwise regression
"""








































