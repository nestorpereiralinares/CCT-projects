# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:46:44 2020

@author: Nestor Pereira
"""

# ref: https://pypi.org/project/pmdarima/1.2.1/
# ref: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
# Ref: W McKinney, 2018, Python for Data Analysis, O'Reilly
# Ref: Data Wrangling, Pag. 221 - 251
# Ref: Data Aggregation and Group Operations, Pag. 287 - 316
# Ref: Time Series, Pag. 317 - 362

# Ref: J Brownlee, 2017, Time Series Forecast Study 
# Ref: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# Ref: J Brownlee, 2017, How to Create an ARIMA Model for Time Series Forecasting,
# Ref: https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
# Ref: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Ref: https://www.statsmodels.org/stable/vector_ar.html#references

# -------------------------
# ML algoritmics Time Serie
# -------------------------

# ARIMA model
#from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

# SARIMA Model
import statsmodels.api as sm
#import pmdarima as pm
#from pmdarima import model_selection

# VAR model
import statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

#seaborn plot 

import seaborn as sns

# matplot lib

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

# seasonal analysis

from pandas import DataFrame
from pandas import Grouper
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------------------------
# require execute: m5_forecasting_dataWrangling.py
# ------------------------------------------------


# ------------------------------------------------
# Descriptive columns Sales
# ------------------------------------------------
#df_sales.head(3).T
#df_sales.dtypes
#df_sales.columns
# Columns:
# 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd',
# 'sales', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
# 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
# 'snap_CA', 'snap_TX', 'snap_WI'


# ------------------------------------------------
# ML algoritmics Time Series
# ------------------------------------------------
#
# Ref: W McKinney, 2018, Python for Data Analysis, O'Reilly
# Ref: Data Wrangling, Pag. 221 - 251
# Ref: Data Aggregation and Group Operations, Pag. 287 - 316
# Ref: Time Series, Pag. 317 - 362
# 

# Datasets
df_sales.columns  # dataset sales
df_cat_sales      # sales for categories
df_cat_food_sales # sales for category Food
df_cat_hobbies_sales # sales for category Hobbies
df_cat_household_sales # sales for category Household


# ------------------------------------------------
# Descriptive statistics
# ------------------------------------------------

print(df_cat_sales.describe())

# ------------------------------------
# BoxPlot by category
# ------------------------------------

#	seaborn styles: 
sns.set_style("darkgrid")
#sns.set_style("dark")

ax = sns.boxplot(data=df_cat_sales,
                 linewidth=0.5,
                 width=0.5,
                 notch=False)

plt.show()

# -------------------
#	Seaborn Histogram 
# -------------------
# Histogram + density grahps

sns.distplot(df_cat_sales["Food"] , color="skyblue")
p1=sns.kdeplot(df_cat_sales['Food'], shade=False, color="b")
p1=sns.kdeplot(df_cat_sales['Hobbies'], shade=False, color="brown")
sns.distplot(df_cat_sales["Hobbies"] , color="brown")
sns.distplot(df_cat_sales["Household"] , color="green")
p1=sns.kdeplot(df_cat_sales['Household'], shade=False, color="g")

plt.xlabel('Total sales')
plt.title('Total sales by Category of products')
plt.show()





# ------------------------------------------------
# Graphs descriptive: seasonal plot
# ------------------------------------------------

# -----------------------------------------
# pre-procesing data for seasonal decompose
# -----------------------------------------

# Ref: J. Chow, 2019, Using the Pandas “Resample” Function, towards data science, 
# Ref: https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4



# data montly
df_cat_food_sales_monthly = df_cat_food_sales.resample('M').sum()[1:-1]  
df_cat_hobbies_sales_monthly = df_cat_hobbies_sales.resample('M').sum()[1:-1]  
df_cat_household_sales_monthly = df_cat_household_sales.resample('M').sum()[1:-1]  

df_cat_food_sales_monthly
df_cat_hobbies_sales_monthly
df_cat_household_sales_monthly

result = seasonal_decompose(np.array(df_cat_food_sales_monthly),model="additive",period=6) # The frequency is semestral
result.plot()

result = seasonal_decompose(np.array(df_cat_hobbies_sales_monthly),period=6) # The frequency is semestral
result.plot()

result = seasonal_decompose(np.array(df_cat_household_sales_monthly),period=6) # The frequency is semestral
result.plot()



# clean and free memory

del df_cat_food_sales_monthly
del df_cat_hobbies_sales_monthly
del df_cat_household_sales_monthly

#gc.collect()


# --------------------------
# Stationarity test
# -------------------------
# Ho: time series is not stationary
# Ha: time series is stationary

# Use the statistical test called Augmented Dickey-Fuller Test (ADF Test).
# The null hypothesis is rejected if p−value<0.05; it suggests thet the time
# series is stationary ( not evidence thet it is not stationary ).

# Augmented Dickey-Fuller test
result = adfuller(df_cat_sales['Food'], autolag='AIC')
print('ADF Statistic for Food: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(df_cat_sales['Hobbies'], autolag='AIC')
print('ADF Statistic for Hobbies: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(df_cat_sales['Household'], autolag='AIC')
print('ADF Statistic for Household: %f' % result[0])
print('p-value: %f' % result[1])

# Fail to reject H0, therefore, the test statistic indicates that 
# There is no evidence to reject the time series is not stationary


# ---------------------------------------------------
# find the order of differencing (d) in AR model
# The purpose of differencing it to make the time series stationary.
# ---------------------------------------------------

# Daily sales by category: Food, Hobbies, Household

df_cat_sales

# original series
df = df_cat_sales
ax = df.plot()
ax.set_title('Original time series')
plt.show()

# 1st Differencing
df_diff = df_cat_sales.diff().dropna()
ax = df_diff.plot()
ax.set_title('Time series 1st First-order differencing')
plt.show()
df_diff

# Fix the order of differencing as d=1 even though the series is not perfectly 
# stationary (weak stationarity).

# -------------------------------------------
# ACF autocorrelation analysys
# Using the differencing scale required to get a near-stationary time series
# Plot ACF help to find the order q of the MA Moving Average
# Considerer q=1 good enough
# -------------------------------------------


from statsmodels.graphics.tsaplots import plot_acf


fig, axes = plt.subplots(3,1, sharex=True)

#plot_acf(df_cat_food_sales, ax=axes[0,0])
#axes[0,0].set_title('Original acf for Food')
plot_acf(df_cat_food_sales.diff(1).dropna(), ax=axes[0])
axes[0].set_title('Autocorrelation 1st differencing acf for Food')

#plot_acf(df_cat_hobbies_sales, ax=axes[1,0])
#axes[1,0].set_title('Original acf for Hobbies')
plot_acf(df_cat_hobbies_sales.diff().dropna(), ax=axes[1])
axes[1].set_title('Autocorrelation 1st differencing acf for Hobbies')

#plot_acf(df_cat_household_sales, ax=axes[2,0])
#axes[2,0].set_title('Original acf for Household')
plot_acf(df_cat_household_sales.diff().dropna(), ax=axes[2])
axes[2].set_title('Autocorrelation 1st differencing acf for Household')

plt.show()

# -------------------------------------------
# Partial autocorrelation PACF analysis
# Using the differencing scale required to get a near-stationary time series
# Plot ACF help to find the order p of the lag for AR autoregressive model
# Considerer p=1 good enough
# -------------------------------------------


from statsmodels.graphics.tsaplots import plot_pacf


fig, axes = plt.subplots(3,1, sharex=True)

plot_acf(df_cat_food_sales.diff(1).dropna(), ax=axes[0])
axes[0].set_title('Partial autocorrelation 1st differencing pacf for Food')

plot_acf(df_cat_hobbies_sales.diff().dropna(), ax=axes[1])
axes[1].set_title('Partial autocorrelation 1st differencing pacf for Hobbies')

plot_acf(df_cat_household_sales.diff().dropna(), ax=axes[2])
axes[2].set_title('Partial autocorrelation 1st differencing pacf for Household')

plt.show()


# Fix the order of differencing as 1 even though the series is not perfectly 
# stationary (weak stationarity).

# ------------------------------
# Correlation between categories
# ------------------------------
corr = df_cat_sales.corr()
print('Correlation: ', corr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_cat_sales.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_cat_sales.columns)
ax.set_yticklabels(df_cat_sales.columns)
plt.show()


# -----------------------------------------------------
# Arima model with seasonality: SARIMA
# -----------------------------------------------------

# -------------------------------------
# Analysis seasonal
# -------------------------------------

# ------------------
# Plot category Food
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(df_cat_food_sales, label='Original data for Food')
axes[0].plot(df_cat_food_sales.diff(1), label='1st Differencing', color='orange')
axes[0].set_title('1st Differencing')
axes[0].legend(loc='upper left', fontsize=10)
# Seasinal Dei
axes[1].plot(df_cat_food_sales, label='Original data for Food')
axes[1].plot(df_cat_food_sales.diff(12), label='Seasonal Differencing', color='orange')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Food sales', fontsize=16)
plt.show()



# ---------------------
# Plot category Hobbies
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(df_cat_hobbies_sales, label='Original data for Hobbies', color='green')
axes[0].plot(df_cat_hobbies_sales.diff(1), label='1st Differencing', color='orange')
axes[0].set_title('1st Differencing')
axes[0].legend(loc='upper left', fontsize=10)
# Seasinal Dei
axes[1].plot(df_cat_hobbies_sales, label='Original data for Hobbies', color='green')
axes[1].plot(df_cat_hobbies_sales.diff(12), label='Seasonal Differencing', color='orange')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Hobbies sales', fontsize=16)
plt.show()



# -----------------------
# Plot category HouseHold
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(df_cat_household_sales, label='Original data for Household', color='brown')
axes[0].plot(df_cat_household_sales.diff(1), label='1st Differencing', color='orange')
axes[0].set_title('1st Differencing')
axes[0].legend(loc='upper left', fontsize=10)
# Seasinal Dei
axes[1].plot(df_cat_household_sales, label='Original data for Household', color='brown')
axes[1].plot(df_cat_household_sales.diff(12), label='Seasonal Differencing', color='orange')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Household sales', fontsize=16)
plt.show()



# -----------------------------------------------------
# Seasonal ARIMA model allows seasonality for the statsmodels SARIMAX
# -----------------------------------------------------
# ref: https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html#examples-notebooks-generated-statespace-sarimax-stata--page-root
# red; https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html


# ------------------------------------------------
# Pre-procesing the data to ML algoritmics Time Series: SARIMAX
# ------------------------------------------------
# -------------------------------------------------
# Datasets sales for category ready for Ml Forecast
# -------------------------------------------------

df_cat_food_sales
df_cat_hobbies_sales
df_cat_household_sales

df_cat_sales # sales for the cat id Food, Hobbies, Household


# --------------------------
# --------------------------
# Analysis for category Food
# --------------------------
# --------------------------

# creating the training set
# the SARIMAX model will be training with all data for dataset

X = df_cat_food_sales
X.index

# --------------------------------------
# fit the model
# SARIMAX (seasonal ARIMA model) support the seasonal effects therefore the original dataset
# without transformation (1st differencing) will be used
# --------------------------------------

import statsmodels.api as sm
# Fit with training samples
seasonalmodel = sm.tsa.statespace.SARIMAX(X,  
                                          order=(2,1,1), 
#(Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity). 
                                          seasonal_order=(2,1,1,12), 
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
                                          

# After try and test, it will found the best model: Fit SARIMA: (2, 1, 1)x(2, 1, 1, 12)
# produce AIC value 34606.97


results = seasonalmodel.fit()
print(results.summary())

#  The P Values of the AR's and MA's terms are highly significant (< 0.05)
#  Therefore, all features will be used.


# ---------------------------------
# Evaluate the model
# ---------------------------------

results.resid.plot()

#  The line plot of the residual errors suggests that almost all information is captured by the model

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# According to these results:
# 1- the KDE line follows closely with the Normal distrubution(0,1) line
# 2- the qq plot show a right line very close to normal
# 3- the line plot of the residual errors suggests that almost all information is captured by the model
# 4- the correlation line shows that the time series residuals have low correlation 

# Therefore, this model is good enough

# --------------------------------
# Compare prediction vs observed data
# --------------------------------

# Create data for test and evaluate the model
X # dataset with all data
X_test = df_cat_food_sales.tail(180) # values observed 180 days
Y_test = df_cat_food_sales.tail(60)  # values to predict and compare 

prediction = results.get_prediction(start='2016-02-25', stop='2016-04-24', dynamic=False)
prediction_ci = prediction.conf_int()
prediction_ci.head()

# --------------------------------
# Plot sales predicted vs observed
# --------------------------------
ax = Y_test.plot(label='Sales observed')
prediction.predicted_mean.plot(ax=ax, label='Sales prediction for 60 days', color='green', linewidth=1)
ax.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='grey', alpha=.5)

ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
ax.set_title('Food sales prediction')
plt.legend()
plt.show()

# ---------------------------------
# Calculate and compare MSE 
# ---------------------------------
mse_prediction = prediction.predicted_mean
mse_test = Y_test

# Compute the mean square error
mse = ((mse_prediction - mse_test) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forecast: {:.4f}'
      .format(np.sqrt(sum((mse_prediction-mse_test)**2)/len(mse_prediction))))


# --------------------------------------
# Forecast for the next 28 days
# --------------------------------------
X  # dataset 
n_periods = 28

forecast = results.get_forecast(steps=(28))
forecast_mean = forecast.predicted_mean
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
# index of the forecast (future dates)
index_of_forecast = pd.date_range(Y_test.index[-1], periods = n_periods, freq='D')
forecast_mean.index = index_of_forecast
forecast_ci.index = index_of_forecast

# Plot
plt.plot(Y_test)
plt.plot(forecast_mean, color='darkgreen', linewidth=1)
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='k', alpha=.15)
ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
plt.legend()
plt.title("Forecast for Food Sales")
plt.show()




# --------------------------
# --------------------------
# Analysis for category Hobbies
# --------------------------
# --------------------------


# creating the training set
# the SARIMAX model will be training with all data for dataset

X = df_cat_hobbies_sales
X.index

# --------------------------------------
# fit the model
# SARIMAX (seasonal ARIMA model) support the seasonal effects therefore the original dataset
# without transformation (1st differencing) will be used
# --------------------------------------

import statsmodels.api as sm
# Fit with training samples
seasonalmodel = sm.tsa.statespace.SARIMAX(X,  
                                          order=(2,1,1), 
#(Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity). 
                                          seasonal_order=(2,1,1,12), 
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
                                          

# After try and test, it will found the best model: Fit SARIMA: (2, 1, 1)x(2, 1, 1, 12)



results = seasonalmodel.fit()
print(results.summary())

#  The P Values of the AR's and MA's terms are highly significant (< 0.05)
#  Therefore, all features will be used.


# ---------------------------------
# Evaluate the model
# ---------------------------------

results.resid.plot()

#  The line plot of the residual errors suggests that almost all information is captured by the model

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# According to these results:
# 1- the KDE line follows closely with the Normal distrubution(0,1) line
# 2- the qq plot show a right line very close to normal
# 3- the line plot of the residual errors suggests that almost all information is captured by the model
# 4- the correlation line shows that the time series residuals have low correlation 

# Therefore, this model is good enough

# --------------------------------
# Compare prediction vs observed data
# --------------------------------

# Create data for test and evaluate the model
X # dataset with all data
X_test = df_cat_hobbies_sales.tail(180) # values observed 180 dyas
Y_test = df_cat_hobbies_sales.tail(60)  # values to predict and compare 

prediction = results.get_prediction(start='2016-02-25', stop='2016-04-24', dynamic=False)
prediction_ci = prediction.conf_int()
prediction_ci.head()

# --------------------------------
# Plot sales predicted vs observed
# --------------------------------
ax = Y_test.plot(label='Sales observed')
prediction.predicted_mean.plot(ax=ax, label='Sales prediction for 60 days', color='green', linewidth=1)
ax.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='grey', alpha=.5)

ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
ax.set_title('Hobbies sales prediction')
plt.legend()
plt.show()

# ---------------------------------
# Calculate and compare MSE 
# ---------------------------------
mse_prediction = prediction.predicted_mean
mse_test = Y_test

# Compute the mean square error
mse = ((mse_prediction - mse_test) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forecast: {:.4f}'.format(np.sqrt(sum((mse_prediction-mse_test)**2)/len(mse_prediction))))


# --------------------------------------
# Forecast for the next 28 days
# --------------------------------------
X  # dataset 
n_periods = 28

forecast = results.get_forecast(steps=(28))
forecast_mean = forecast.predicted_mean
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
# index of the forecast (future dates)
index_of_forecast = pd.date_range(X.index[-1], periods = n_periods, freq='D')
forecast_mean.index = index_of_forecast
forecast_ci.index = index_of_forecast

# Plot
plt.plot(Y_test)
plt.plot(forecast_mean, color='darkgreen')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='k', alpha=.15)
ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
plt.legend()
plt.title("Forecast for Hobbies Sales")
plt.show()





# --------------------------
# --------------------------
# Analysis for category Household
# --------------------------
# --------------------------



# creating the training set
# the SARIMAX model will be training with all data for dataset

X = df_cat_household_sales
X.index

# --------------------------------------
# fit the model
# SARIMAX (seasonal ARIMA model) support the seasonal effects therefore the original dataset
# without transformation (1st differencing) will be used
# --------------------------------------

import statsmodels.api as sm
# Fit with training samples
seasonalmodel = sm.tsa.statespace.SARIMAX(X,  
                                          order=(2,1,1), 
#(Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity). 
                                          seasonal_order=(2,1,1,12), 
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
                                          

# After try and test, it will found the best model: Fit SARIMA: (2, 1, 1)x(2, 1, 1, 12)



results = seasonalmodel.fit()
print(results.summary())

#  The P Values of the AR's and MA's terms are highly significant (< 0.05)
#  Therefore, all features will be used.


# ---------------------------------
# Evaluate the model
# ---------------------------------

results.resid.plot()

#  The line plot of the residual errors suggests that almost all information is captured by the model

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# According to these results:
# 1- the KDE line follows closely with the Normal distrubution(0,1) line
# 2- the qq plot show a right line very close to normal
# 3- the line plot of the residual errors suggests that almost all information is captured by the model
# 4- the correlation line shows that the time series residuals have low correlation 

# Therefore, this model is good enough

# --------------------------------
# Compare prediction vs observed data
# --------------------------------

# Create data for test and evaluate the model
X # dataset with all data
X_test = df_cat_household_sales.tail(180) # values observed 180 dyas
Y_test = df_cat_household_sales.tail(60)  # values to predict and compare 

prediction = results.get_prediction(start='2016-02-25', stop='2016-04-24', dynamic=False)
prediction_ci = prediction.conf_int()
prediction_ci.head()

# --------------------------------
# Plot sales predicted vs observed
# --------------------------------
ax = Y_test.plot(label='Sales observed')
prediction.predicted_mean.plot(ax=ax, label='Sales prediction for 60 days', alpha=.7, color='green')
ax.fill_between(prediction_ci.index,
                prediction_ci.iloc[:, 0],
                prediction_ci.iloc[:, 1], color='grey', alpha=.5)

ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
ax.set_title('Household sales prediction')
plt.legend()
plt.show()

# ---------------------------------
# Calculate and compare MSE 
# ---------------------------------
mse_prediction = prediction.predicted_mean
mse_test = Y_test

# Compute the mean square error
mse = ((mse_prediction - mse_test) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forecast: {:.4f}'
      .format(np.sqrt(sum((mse_prediction-mse_test)**2)/len(mse_prediction))))


# --------------------------------------
# Forecast for the next 28 days
# --------------------------------------
X  # dataset 
n_periods = 28
forecast = results.get_forecast(steps=(28))
forecast_mean = forecast.predicted_mean
# Get confidence intervals of forecasts
forecast_ci = forecast.conf_int()
# index of the forecast (future dates)
index_of_forecast = pd.date_range(X.index[-1], periods = n_periods, freq='D')
forecast_mean.index = index_of_forecast
forecast_ci.index = index_of_forecast

# Plot
plt.plot(Y_test)
plt.plot(forecast_mean, color='darkgreen')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='k', alpha=.15)
ax.set_xlabel('Time (daily)')
ax.set_ylabel('Sales')
plt.legend()
plt.title("Forecast for Hobbies Sales")
plt.show()





