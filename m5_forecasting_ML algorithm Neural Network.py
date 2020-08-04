# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:51:06 2020
 
@author: Nestor Pereira
"""
# -------------------------
# ML algoritmics Neural Network 
# Ref: J. Brownlee, 2020, Deep Learning for Time Series Forecasting, Edition: v1.7
# Ref: G. Bontempi, S. Ben, Y. Le, 2013, Machine Learning Strategies for Time Series Forecasting, ResearchGate.net
#    https://www.researchgate.net/publication/236941795_Machine_Learning_Strategies_for_Time_Series_Forecasting
# -------------------------

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import pickle

#  Neural Network Model



# ------------------------------------------------
# Descriptive arrays for Neural Networks
# ------------------------------------------------
food   
hobbies
house

# ------------------------------------------------
# Pre-procesing the data to ML algoritmics Neural Network
# ------------------------------------------------
#
# Ref: W McKinney, 2018, Python for Data Analysis, O'Reilly
# Ref: Data Wrangling, Pag. 221 - 251
# Ref: Data Aggregation and Group Operations, Pag. 287 - 316
# Ref: J Brownlee, 2017, Time Series Forecast Study 
# Ref: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# Ref: J Brownlee, 2020, Deep Learning for Time Series Forecasting, Editionv1.7 
# 
# ------------------------------------------------
# Pre-procesing the daily data 
# by category id: Foods, Hobbies, Household 
# ------------------------------------------------

# -------------------------------------------
# Scaling the data
# -------------------------------------------

# Usually it is necesary scaling the data in forecasting with diferent scale
# and because the time series is not stationary. However, Neural Network 
# algoritm is capable to learn and work from the raw data without scaling.
# the argument of this is because the Neural Network usually is robust to noise
# in the input data and not make strong assumtions about the mapping function, 
# therefore, it is learn form linear and non-linear relationships.


# -------------------------------------------
# Dealing with outliers
# -------------------------------------------
# Define threshold as minimum values
# the data present some values extreme that will be replaced by the mean point
# between the neighbours.

# ------------------------------------------------
# detecting outliers by graphs descriptive: line plot
# ------------------------------------------------
    
plt.figure(figsize=(10,10))
plt.plot(food.ravel(), color='blue')
plt.plot(hobbies.ravel(), color='brown')
plt.plot(house.ravel(), color='green')
plt.title('Daily sales by categories: Food, Hobbies, Household')
plt.show()

# According to the analysis of the graph, it will be used 5.000 
# as the threshold for products Food
ierror=np.argwhere(food<5000)[:,1] # Find the indices of array with outliers
# replace outliers by the mean point between the neighbours
for i in ierror:
    food[:,i]=(food[:,i-1]+food[:,i+1])/2
    
# According to the analysis of the graph, it will be used 1.000 
# as the threshold for products Hobbies
ierror=np.argwhere(hobbies<1000)[:,1] # Find the indices of array with outliers
# replace outliers by the mean point between the neighbours
for i in ierror:
    hobbies[:,i]=(hobbies[:,i-1]+hobbies[:,i+1])/2
    
# According to the analysis of the graph, it will be used 2.500 
# as the threshold for products Household
ierror=np.argwhere(house<2500)[:,1] # Find the indices of array with outliers
# replace outliers by the mean point between the neighbours
for i in ierror:
    house[:,i]=(house[:,i-1]+house[:,i+1])/2

# verify the days using the index:
np.arange(food.shape[1]).reshape(1,food.shape[1])
np.arange(hobbies.shape[1]).reshape(1,hobbies.shape[1])
np.arange(house.shape[1]).reshape(1,house.shape[1])

# -------------------------------------------------------------------
# 30 day ahead product sales prediction based on past ten date sales 
# -------------------------------------------------------------------

def forecast_based_NN(product, name):

# ---------------------------------
# create the Neural Network
# using one hidden layer just by experimentation
# ---------------------------------

    model = Sequential()
# input dimension: feature dimension of the Xtrain/Xtest
    model.add(Dense(60, input_dim=pastdays, activation='relu'))
# one hidden layers
    model.add(Dense(80, activation='relu'))
#    model.add(Dense(60,  activation='relu'))
# number of layers here is the feature dimension of output
    model.add(Dense(futuredays, activation='linear'))

# save a plot of the model

    plot_model(model, show_shapes=True, to_file='multiheaded.png')
    
# Compile model using MSE 
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(Xtrain.T, Ytrain.T, epochs=20, verbose=1)

# -----------
# prediction 
# ----------
    Ypred = model.predict(Xtest.T).T
    
    print('*** Prediction for product: %s' % name )
  

# ------------------------
# save model
# ------------------------
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    #file_model = 'forecast_NN_model.sav'
    #pickle.dump(model, open(file_model, 'wb'))

    
# ----------------------
# Evaluate performance
# ----------------------
# R2 is a commonly used prediction accuracy measure, optimal=1, bad=0
# Evaluate RMSE

    R2=r2_score(Ytest,Ypred)
    print('R-squared: %.2f' % R2)

    rmserror = math.sqrt(mean_squared_error(Ytest,Ypred))
    print('Root Mean Squared Error RMSE: %.2f' % rmserror)

    print(' Data for test: ', Ytest[0],'\n Data predicted: ', Ypred[0])
    
# ----------------------
# Evaluation performance by graphs 
# ----------------------   
    #for i in range(Ypred.shape[1]):
    for i in range(0,28,14):
        plt.figure(figsize=(10,10))
        fromdate=df_cal.iloc[train+i-1,0]
        plt.title("30 days ahead prediction on the day: %s" 
                  % (fromdate))
 #             % (fromdate.strftime("%d-%b-%Y")))
        plt.plot(Ytest[:,i].ravel())
        plt.plot(Ypred[:,i].ravel())
    
    return

# -------------------------------------------------------------------------
# Apply the model for each category of product: Food, Hobbies and Household
# -------------------------------------------------------------------------

# ==================================
# ----------------------------------
# Evaluate for product: Food
#-----------------------------------
# ==================================
product = 'food'


# ------------------------------------
# Create dataset for training and test   
# ------------------------------------

# In time series is common to use the prior time steps of the feature 
# to predict the next step time of the output feature (Lag method). 
# In order to apply neural Network algorithmics, it is necessary to iterate 
# over time steps and divide the data into overlapping windows. 
# In this case, it will iterate over windows of ten days from the input data 
# to predict the subsequent 30 days in the output data. 
# From the input data, it will be used sets of 10 days, overlapping, 
# from the first 1830 days for training and the rest for the test.

pastdays=10 # time steps or lag the past 10 days
futuredays=30 # time steps 30 days for prediction
train=1850  # days taken form the data to train the model

print('Data available for modelling: ', food.shape[1])
print('Days have taken for the data to train the model: ', train)
print('Dayshave taken for time steps or lag from the past: ', pastdays)
print('Days time steps for prediction: ', futuredays)

 # Training dataset
Xtrain=np.zeros((pastdays,train-futuredays))
Ytrain=np.zeros((futuredays,train-futuredays))
# prepare the data from training and test 
for i in range(pastdays,train-futuredays):
    Xtrain[:,i-pastdays]=food[:,i-pastdays:i]
    Ytrain[:,i-pastdays]=food[:,i:i+futuredays]

# Test dataset
Xtest=np.zeros((pastdays,food.shape[1]-futuredays-train))
Ytest=np.zeros((futuredays,food.shape[1]-futuredays-train))

for i in range(train,food.shape[1]-futuredays):
        Xtest[:,i-train]=food[:,i-pastdays:i]
        Ytest[:,i-train]=food[:,i:i+futuredays]   

# Apply the model for the training and test dataset

forecast_based_NN(food, 'Food')  

# --------------------------------------
# Prediction new values from the day 1913
# ---------------------------------------

# Calculate a new data without know the results in oder to predict the next 28 days
# parameters to produce new Xtest with the last data available to produce future predictions

# ---------------------------------
# load model
# ---------------------------------
filename = 'finalized_model.sav'
nn_model = pickle.load(open(filename, 'rb'))
# ----------------------------------
# Apply the model NN for forecasting
# ----------------------------------

# ------------------------------------------------------------
# Predict the 28 days unknowed using  the last five days known
# ------------------------------------------------------------
# Test dataset using the last 5 values
lastindex=1913-14; lastindex
newXtest=np.zeros((10,5))
for i in range(0,5):
    newXtest[:,i]=food[:,lastindex+i:lastindex+i+10]
    print(newXtest[:,i])
    print('food:', food[:,lastindex+i:lastindex+i+10])

# Apply the model to the new data to predict 28 days  
newYpred = nn_model.predict(newXtest.T).T
# Plot the graph using the last data know to predict the next 28 days from the last day.
#for i in range(Ypred.shape[1]):
index_date_start=1901
for i in range(1):  
    plt.figure(figsize=(10,10))
    fromdate=df_cal.iloc[index_date_start,0]
    plt.title("30 days ahead prediction Food on the day: %s" 
                  % (fromdate))
    plt.plot(newXtest[:,i+3].ravel())
    plt.plot(newYpred[:,i].ravel())



# -------------------------------------------------------------------------
# Apply the model for each category of product: Food, Hobbies and Household
# -------------------------------------------------------------------------

# ==================================
# ----------------------------------
# Evaluate for product: Hobbies
#-----------------------------------
# ==================================
product = 'hobbies'

# ------------------------------------
# Create dataset for training and test   
# ------------------------------------

# From the input data, it will be used sets of 10 days, overlapping, 
# from the first 1830 days for training and the rest for the test.

pastdays=10 # time steps or lag the past 10 days
futuredays=30 # time steps 30 days for prediction
train=1850  # days taken form the data to train the model

print('Data available for modeling: ', hobbies.shape[1])
print('Days taken for the data to train the model: ', train)
print('Days taken for time steps or lag from the past: ', pastdays)
print('Days time steps for prediction: ', futuredays)

 # Training dataset
Xtrain=np.zeros((pastdays,train-futuredays))
Ytrain=np.zeros((futuredays,train-futuredays))
# prepare the data from training and test 
for i in range(pastdays,train-futuredays):
    Xtrain[:,i-pastdays]=hobbies[:,i-pastdays:i]
    Ytrain[:,i-pastdays]=hobbies[:,i:i+futuredays]

# Test dataset
Xtest=np.zeros((pastdays,hobbies.shape[1]-futuredays-train))
Ytest=np.zeros((futuredays,hobbies.shape[1]-futuredays-train))

for i in range(train,hobbies.shape[1]-futuredays):
        Xtest[:,i-train]=hobbies[:,i-pastdays:i]
        Ytest[:,i-train]=hobbies[:,i:i+futuredays]   

# Apply the model for the training and test dataset

forecast_based_NN(hobbies, 'Hobbies')  

# --------------------------------------
# Prediction new values from the day 1913
# ---------------------------------------

# Calculate a new data without know the results in oder to predict the next 28 days
# parameters to produce new Xtest with the last data available to produce future predictions

# ---------------------------------
# load model
# ---------------------------------
filename = 'finalized_model.sav'
nn_model = pickle.load(open(filename, 'rb'))
# ----------------------------------
# Apply the model NN for forecasting
# ----------------------------------


# ------------------------------------------------------------
# Predict the 28 days unknowed using  the last five days known
# ------------------------------------------------------------
# Test dataset using the last 5 values
lastindex=1913-14; lastindex
newXtest=np.zeros((10,5))
for i in range(0,5):
    newXtest[:,i]=hobbies[:,lastindex+i:lastindex+i+10]
    print(newXtest[:,i])
    print('hobbies:', hobbies[:,lastindex+i:lastindex+i+10])

# Apply the model to the new data to predict 28 days  
newYpred = nn_model.predict(newXtest.T).T
# Plot the graph using the last data know to predict the next 28 days from the last day.
#for i in range(Ypred.shape[1]):
index_date_start=1901
for i in range(1):  
    plt.figure(figsize=(10,10))
    fromdate=df_cal.iloc[index_date_start,0]
    plt.title("30 days ahead prediction Hobbies on the day: %s" 
                  % (fromdate))
    plt.plot(newXtest[:,i+3].ravel())

    plt.plot(newYpred[:,i].ravel())




# -------------------------------------------------------------------------
# Apply the model for each category of product: Food, Hobbies and Household
# -------------------------------------------------------------------------

# ==================================
# ----------------------------------
# Evaluate for product: Household
#-----------------------------------
# ==================================
product = 'house'

# ------------------------------------
# Create dataset for training and test   
# ------------------------------------

# In time series is common to use the prior time steps of the feature 
# to predict the next step time of the output feature (Lag method). 
# In order to apply neural Network algorithmics, is necessary to iterate 
# over time steps and divide the data into overlapping windows. 
# In this case, it will iterate over windows of ten days from the input data 
# to predict the subsequent 30 days in the output data. 
# From the input data, it will be used sets of 10 days, overlapping, 
# from the first 1830 days for training and the rest for the test.

pastdays=10 # time steps or lag the past 10 days
futuredays=30 # time steps 30 days for prediction
train=1850  # days taken form the data to train the model

print('Data available for modeling: ', house.shape[1])
print('Days taken for the data to train the model: ', train)
print('Days taken for time steps or lag from the past: ', pastdays)
print('Days time steps for prediction: ', futuredays)

 # Training dataset
Xtrain=np.zeros((pastdays,train-futuredays))
Ytrain=np.zeros((futuredays,train-futuredays))
# prepare the data from training and test 
for i in range(pastdays,train-futuredays):
    Xtrain[:,i-pastdays]=house[:,i-pastdays:i]
    Ytrain[:,i-pastdays]=house[:,i:i+futuredays]

# Test dataset
Xtest=np.zeros((pastdays,house.shape[1]-futuredays-train))
Ytest=np.zeros((futuredays,house.shape[1]-futuredays-train))

for i in range(train,house.shape[1]-futuredays):
        Xtest[:,i-train]=house[:,i-pastdays:i]
        Ytest[:,i-train]=house[:,i:i+futuredays]   

# Apply the model for the training and test dataset

forecast_based_NN(house, 'Household')  

# --------------------------------------
# Prediction new values from the day 1913
# ---------------------------------------

# Calculate a new data without know the results in oder to predict the next 28 days
# parameters to produce new Xtest with the last data available to produce future predictions

# ---------------------------------
# load model
# ---------------------------------
filename = 'finalized_model.sav'
nn_model = pickle.load(open(filename, 'rb'))
# ----------------------------------
# Apply the model NN for forecasting
# ----------------------------------


# ------------------------------------------------------------
# Predict the 28 days unknowed using  the last five days known
# ------------------------------------------------------------
# Create a Test dataset using the last 5 values

#
lastindex=1913-14; lastindex
newXtest=np.zeros((10,5))
for i in range(0,5):
    newXtest[:,i]=house[:,lastindex+i:lastindex+i+10]
    print(newXtest[:,i])
    print('household:', house[:,lastindex+i:lastindex+i+10])

# Apply the model to the new data to predict 28 days  
newYpred = nn_model.predict(newXtest.T).T
# Plot the graph using the last data know to predict the next 28 days from the last day.
#for i in range(Ypred.shape[1]):
index_date_start=1901
for i in range(1):  
    plt.figure(figsize=(10,10))
    fromdate=df_cal.iloc[index_date_start,0]
    plt.title("30 days ahead prediction Household on the day: %s" 
                  % (fromdate))
    plt.plot(newXtest[:,i+3].ravel())
    plt.plot(newYpred[:,i].ravel())








# -------------------------------------------------------
# --------- end Neural Network --------------------------
# -------------------------------------------------------


