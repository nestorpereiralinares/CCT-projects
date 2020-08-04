# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:46:44 2020

@author: Nestor Pereira
"""

# Ref: W McKinney, 2018, Python for Data Analysis, O'Reilly
# Ref: Data Wrangling, Pag. 221 - 251
# Ref: Data Aggregation and Group Operations, Pag. 287 - 316
# Ref: Time Series, Pag. 317 - 362

# Ref: J Brownlee, 2017, Time Series Forecast Study 
# Ref: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# Ref: J Brownlee, 2017, How to Create an ARIMA Model for Time Series Forecasting,
# Ref: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Ref: https://www.statsmodels.org/stable/vector_ar.html#references

# -------------------------
# ML algoritmics Time Serie VAR
# -------------------------

# ARIMA model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

# SARIMA Model
import statsmodels.api as sm

# VAR model
import statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

#seaborn plot 

import seaborn as sns

# seasonal analysis

from pandas import DataFrame
from pandas import Grouper
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------------------------
# require execute: m5_forecasting_dataWrangling.py
# ------------------------------------------------

#import m5_forecasting_dataWrangling

# --------------------------------------------------------------
# VAR Model applying for categories: food, hobbies and household
# --------------------------------------------------------------

# Vector Autoregression (VAR) is a multivariate forecasting algorithm 
# that is used when two or more time series influence each other.
# According to the correlation test there are some relationships between Food,
# Hobbies and Household products. However, it is difficult to define which 
# product are related to each other.


# Since the VAR model requires the time series are stationary, therefore, 
# the original data has to be transformer using the 1st differencing.


# ------------------------------------------------
# Pre-procesing the data to ML algoritmics Time Series: VAR Model
# ------------------------------------------------

#If Y(t) is the value at time t, then the first difference is #Y'(t) = Y(t) - Y(t-1). The differencing of the series is #subtracting the current value by the previous value.
# Update df differencing to apply VAR model

df_cat_sales
df_cat_sales_diff = df_cat_sales.diff().dropna()

df_cat_sales_diff # using 1st Differences Dataframe

# test back
df_test = df_cat_sales_diff.copy()

#creating the train and validation set
# -------------
# split dataset
# -------------

# ---------------------------------
nobs = 30 # predictions for 30 days
# ---------------------------------
X_train_diff = df_cat_sales_diff[0:-nobs]  # pass observations data for training
X_test_diff = df_cat_sales_diff[-nobs:]    # 30 days for test

X_train = df_cat_sales[0:-nobs]  # original data for training
X_train
X_test = df_cat_sales[-nobs:]  # data for 30 days of prediction
X_test

# data with differencing
X_train_diff
X_test_diff # 30 days of observations for test the model


# The VAR class assumes that the passed time series are stationary.

# ---------------------------------------
# Select the order (P) of VAR model
# ---------------------------------------

model = VAR(X_train_diff)
for i in range(1,13):
    result = model.fit(i)
    print('Lag Order =', i, 'AIC : ', result.aic)
    
# the AIC drops to lowest at lag 9
P = 2 

# ----------------------------
# Apply VAR model
# ----------------------------

model = VAR(endog=X_train_diff) # training using 1st Differences Dataframe
results = model.fit(P) # Maximum number of lags 9 to check for order selection
results.plot()
results.summary()


# Verify the lag order 
lag_order = results.k_ar
print(lag_order)

# input data for forescasting

#forecast_input = X_train.values[-lag_order:]
forecast_input = X_train_diff.values[-nobs:]
forecast_input.shape

# Forecast, make prediction on validation

# Forecast
# nobs observations in test file ( prediction for 30 days )
fc = results.forecast(forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=X_test.index[-nobs:], columns=X_test_diff.columns)
#df_forecast = pd.DataFrame(fc, index=X_test.index[-nobs:], columns=X_test.columns + '_1d')
df_forecast

# --------------------------------------------------
# Invert the transformation to get the real forecast
# --------------------------------------------------
    
columns = df_cat_sales.columns
columns
 
# --------------------------------------------------
# Invert the transformation to get the real forecast
# --------------------------------------------------

df_results = df_forecast.copy()

for col in columns:        
        # Roll back 1st Differencing
        df_results[str(col)+'_forecast'] = X_test[col].iloc[-30:].values + df_forecast[str(col)].iloc[-30:].cumsum()
        #df_results[str(col)+'_forecast'] = (X_test[col].iloc[-30:]-X_test_diff[col].iloc[-30:]) + df_forecast[str(col)].cumsum()
         

df_results = df_results.loc[:,['Food_forecast', 'Hobbies_forecast', 'Household_forecast']]

# 30 days prediction series
ax = df_results.plot()
ax.set_title('Prediction 30 days for daily sales by categories: Food, Hobbies, Household')
plt.show()


# ---------------------------------
# Plot of Forecast vs actual values
# ---------------------------------


fig = plt.figure()
ax = X_test.plot(label='Sales actuals')
df_results.plot(ax=ax, label='Sales prediction for 30 days', color='r', linestyle='dotted', linewidth=1)
ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
ax.set_title('Sales: Forecast vs Actuals')
plt.legend(fontsize=7, loc=1)
plt.show()



# -------------------------------
# Evaluate the performance RMSE
# -------------------------------

X_test
df_results

# comprative values rmse
for i in columns:
    print('RMSE value for', i, 'is : ', sqrt(mean_squared_error(df_results[i+'_forecast'], X_test[i])))









# -----------------------------------------------
# VAR Model: Training the model for all products
# -----------------------------------------------

# ====================================================
# ----------------------------------------------------
# expand the VAR model to all products
# apply to df submissions diff over all dataset
# Evaluate performance
# ----------------------------------------------------
# ====================================================

# ------------------------------------------------
# Pre-procesing the data 
# ------------------------------------------------

df_sales_all = df_sales[['id', 'd', 'sales', 'date']]
# Pivot and reset the index date
df_sales_all = df_sales_all.pivot(index='date', columns='id', values='sales')
df_sales_all



# Time series 1st First-order differencing
# Daily sales for all products: Food, Hobbies, Household

# ---------------------------------------------------
# find the order of differencing (d=1) in AR model
# The purpose of differencing it to make the time series stationary.
# ---------------------------------------------------

# ----------------
# 1st Differencing
# ----------------
df_diff = df_sales_all.diff().dropna()

# Daily sales by all products in: Food, Hobbies, Household

df_diff

# -----------------------------------------------
# VAR Model 
# -----------------------------------------------
#creating the train and validation set from 1st differencing data
# -------------
# using complete dataset
# -------------

# ---------------------------------
nobs = 28 # predictions for 28 days
# ---------------------------------


X_train = df_sales_all.copy()
X_train_diff = df_diff.copy()

X_test = X_train[-nobs:]
X_test_diff = X_train_diff[-nobs:]

X_train = X_train[0:-nobs]
X_train_diff = X_train_diff[0:-nobs]

# reduce the size of the sample
X_train = X_train[1850:]  # 60 days pass observations data for training
X_train_diff = X_train_diff[1850:]  # 60 days pass observations data for training

X_train
X_train_diff

X_test
X_test_diff

# --------------------------------------
# fit the model
# --------------------------------------


# -------------------------------------------------
# Training the model for prediction
# Final prediction for all features (products) df submissions
# -------------------------------------------------


# The VAR class assumes that the passed time series are stationary.

# ---------------------------------------
# Select the order (P) of VAR model
# ---------------------------------------
# 
P = 1 
# ----------------------------
# Apply VAR model
# ----------------------------


model = VAR(endog=X_train_diff) # training using 1st Differences Dataframe
results = model.fit(P) # lag number p

# Verify the lag order 
lag_order = results.k_ar
print(lag_order)

X_train_diff[-lag_order:]

# input data for forescasting for the original data (not need use the 1st differencing)


forecast_input = X_train_diff.values[-lag_order:]
forecast_input
X_train_diff[-lag_order:]
columns = X_train_diff[-lag_order:].columns
columns
# Forescast, make prediction based on the last original data 

# Forecast
# nobs observations in test file ( prediction for 28 days )

fc = results.forecast(forecast_input, steps=nobs) # nobs = 28 days
df_forecast = pd.DataFrame(fc, 
                           index=X_test.index, 
                           columns=columns)

df_forecast


# --------------------------------------------------
# Invert the transformation to get the real forecast
# --------------------------------------------------

df_results = df_forecast.copy()


for col in columns:
        df_forecast[col].iloc[0] = df_forecast[col].iloc[0] + X_train[col].iloc[-2]
        # invert transformation 1st Diff
        for i in range(1,27):
            df_forecast[col].iloc[i] = df_forecast[col].iloc[i] + df_forecast[col].iloc[i-1]

# Because the forecast is made based on differencing transformation, 
# the process to back to the original scale on new data generated by the model
# produce negative values that will be considered "noise". Therefore, it will be
# substitute by zero. It is a common practice.
            
# However, it means the model VAR is not good enough for large numbers of feature
# perhaps because it is a not time series stationary and also the dependency 
# between the features is not clear and difficult to prove. 
         
df_forecast[df_forecast < 0] = 0

df_forecast  # final prediction 

# -------------------------------
# Evaluate the performance RMSE
# -------------------------------

X_test
df_forecast

RMSE_values = pd.DataFrame(forecast_input,
                           columns=columns)
# comparative values rmse
for i in columns:
    RMSE_values[i] = sqrt(mean_squared_error(df_forecast[i], X_test[i]))

print('RMSE mean value is: ', RMSE_values.mean().mean())


# -------------------------------
# Plot results (sample)
# -------------------------------

# 28 days prediction series
fig = plt.figure()
df = df_forecast[['HOBBIES_1_009_CA_1_validation','HOUSEHOLD_2_516_TX_1_validation','FOODS_3_174_CA_4_validation']]
ax = df.plot()
ax.set_title('Prediction 28 days for daily sales by products: sample of Food, Hobbies, Household')
plt.legend(fontsize=7, loc=1)
plt.show()



# ====================================================
# ----------------------------------------------------
# Making final predictions 
# Validation data
# apply to df submissions Using diff
# ----------------------------------------------------
# ====================================================

del df_cal
del df_sales_train
del df_prices
del df_cat_sales
del df_cat_food_sales
del df_cat_hobbies_sales
del df_cat_household_sales

import gc
gc.collect()




# VAR model
import statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic


# verify dataset submission


df_submissions.head(5)
df_submissions.tail(10)

df_submissions_predictions # dataframe ready

# ------------------------------------------------
# Pre-procesing the data 
# ------------------------------------------------

df_sales_all = df_sales[['id', 'd', 'sales', 'date']]
# Pivot and reset the index date
df_sales_all = df_sales_all.pivot(index='date', columns='id', values='sales')
df_sales_all.shape

df_submissions_all.head(5)
df_submissions_all.tail(10).T

df_submissions_predictions
df_submissions_evaluation



# Time series 1st First-order differencing
# Daily sales for all products

# ---------------------------------------------------
# find the order of differencing (d=1) in AR model
# The purpose of differencing it to make the time series stationary.
# ---------------------------------------------------

# ----------------
# 1st Differencing
# ----------------
df_diff = df_sales_all.diff().dropna()

# Daily sales by all products in: Food, Hobbies, Household

df_diff

# -----------------------------------------------
# VAR Model 
# -----------------------------------------------

#creating the train and validation set from 1st differencing data
# -------------
# using complete dataset
# -------------

# ---------------------------------
nobs = 28 # predictions for 28 days for validation (d1914 to d1941)
nobs_evaluation = 28 # for future evaluation predictions for 28 days for evaluation (d1942 to d1969)
# ---------------------------------



X_train = df_sales_all.copy()
X_train_diff = df_diff.copy()



X_train = X_train[1850:]  # 60 days pass observations data for training
X_train_diff = X_train_diff[1850:]  # 60 days pass observations data for training

X_train
X_train_diff.shape


# -------------------------------------------------
# Training the model for prediction
# Final prediction for all features (products) df submissions
# -------------------------------------------------

# --------------------------------------
# fit the model
# --------------------------------------

# The VAR class assumes that the passed time series are stationary.

# ---------------------------------------
# Select the order (P) of VAR model
# ---------------------------------------
# 
P = 1 
# ----------------------------
# Apply VAR model
# ----------------------------


model = VAR(endog=X_train_diff) # training using 1st Differences Dataframe
results = model.fit(P) # lag number p



# Verify the lag order 
lag_order = results.k_ar
print(lag_order)


X_train[-lag_order:]
df_sales_all[-lag_order:]


# input data for forescasting for the original data (not need use the 1st differencing)


forecast_input = X_train.values[-lag_order:]
forecast_input
X_train[-lag_order:]
columns = X_train[-lag_order:].columns
columns
# Forescast, make prediction based on the last original data 
# Reorder the columns on dataframe

df_submissions_predictions.columns = columns
df_submissions_evaluation.columns = columns
columns


# Forecast
# nobs observations in test file ( prediction for 28 days )

fc = results.forecast(forecast_input, steps=(nobs)) # nobs = 28 days (28 days)
#fc = results.forecast(forecast_input, steps=(nobs+nobs_evaluation)) # nobs = 28 days + nobs_evaluation (28 days)
df_forecast = pd.DataFrame(fc, 
                           #index=df_submissions_predictions.index, 
                           columns=columns)

df_forecast




# --------------------------------------------------
# Invert the transformation to get the real forecast
# --------------------------------------------------

df_results = df_forecast.copy()



for col in columns:
        df_forecast[col].iloc[0] = df_forecast[col].iloc[0] + X_train[col].iloc[-2]
        # invert transformation 1st Diff
        for i in range(1,27):
            df_forecast[col].iloc[i] = df_forecast[col].iloc[i] + df_forecast[col].iloc[i-1]

# Because the forecast is made based on differencing transformation, 
# the process to back to the original scale on new data generated by the model
# produce negative values that will be considered "noise". Therefore, it will be
# substitute by zero. It is a common practice.
            
# However, it means the model VAR is not good enough for large numbers of feature
# perhaps because it is a not time series stationary and also the dependency 
# between the features is not clear and difficult to prove. 
         
df_forecast[df_forecast < 0] = 0

df_forecast  # final prediction 

# -------------------------------
# Plot results (sample)
# -------------------------------

# 28 days prediction series
df = df_forecast[['HOBBIES_1_009_CA_1_validation', 'HOUSEHOLD_1_370_TX_3_validation', 'FOODS_3_174_CA_4_validation']]
ax = df.plot()
ax.set_title('Prediction 28 days for daily sales by products: sample of Food, Hobbies, Household')
plt.show()


# ---------------------
# ---------------------
# prepare final dataset
# ---------------------
# ---------------------
df2

df1 = df_forecast.iloc[0:28].copy() # forecast d1914 to d1941
df2 = df_forecast.iloc[0:28].copy()# future evaluation
#df2 = df_forecast.iloc[28:].copy()  # future evaluation forecast d1942 to d1969
# df2 will be properly calculated one month before the finish date of competition

df1 = pd.DataFrame(df_forecast.iloc[0:28].values, # forecast d1914 to d1941
                   index=df_submissions_predictions.index, 
                   columns=df_submissions_predictions.columns)


df2 = pd.DataFrame(df_forecast.iloc[0:28].values, # forecast d1914 to d1941
                   index=df_submissions_evaluation.index, 
                   columns=df_submissions_evaluation.columns)



#df2 = pd.DataFrame(df_forecast.iloc[28:].values, # forecast d1942 to d1969
#                   index=df_submissions_evaluation.index, 
#                   columns=df_submissions_evaluation.columns)


# update df submissions with the results

df_submissions_predictions = df1 # forecast d1914 to d1941


df_submissions_evaluation = df2 # future evaluation forecast d1942 to d1969

# traspose the dataframe and add columns id
# ----------------------------------------------
# dataframe validation - forecast d1914 to d1941
# ----------------------------------------------
df1 = df_submissions_predictions.T # traspose 

df1.reset_index(level=0, inplace=True) # add columns id

df_submissions_predictions = df1 # forecast d1914 to d1941
# ----------------------------------------------
# dataframe evaluation - forecast d1942 to d1969
# ----------------------------------------------
df2 = df_submissions_evaluation.T # traspose 

df2.reset_index(level=0, inplace=True) # add columns id

df2.id = df2.id.str.replace("validation", "evaluation") # replace validation to evaluation

df_submissions_evaluation = df2.copy()

# ---------------------------
# update df submissions
# ---------------------------

df_submissions_back = df_submissions.copy()  # dataframe ready

df_submissions = pd.concat([df1, df2])



# Save the df_submissions as csv file ready to send
csv_submit = df_submissions.to_csv('submission_final_NPereira.csv', index = False)


# ----------------------
# end VAR model
# ----------------------

