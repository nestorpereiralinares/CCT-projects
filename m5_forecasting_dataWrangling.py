# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:01:22 2020

Data Wrangling 


Scope:
    
    The objective of this project is to produce the forecast for the next 28 days 
    for the total of daily sales of the categories: Food, Hobbies and Household. 
    The original competition from Kaggle asks for a prediction for the next 28 days 
    of the unit sales of various products sold, inside of these categories,  
    in the USA by Walmart. 
    https://www.kaggle.com/c/m5-forecasting-accuracy/overview/description


@author: Nestor Pereira
"""
# import functions
import pandas as pd
import numpy as np
from pandas import read_csv

# for data Wrangling

# metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

# matplot lib

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})

#seaborn plot 

import seaborn as sns

# pickle to save file
import pickle

# -------------------------
# ML algoritmics Time Serie
# -------------------------

# ARIMA model
#from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.arima_model import ARIMAResults

# VAR model
# Import Statsmodels
#from statsmodels.tsa.api import VAR
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tools.eval_measures import rmse, aic

# --------------------------
# reference:
# --------------------------

# Ref: W McKinney, 2018, Python for Data Analysis, O'Reilly
# Ref: Data Wrangling, Pag. 221 - 251
# Ref: Data Aggregation and Group Operations, Pag. 287 - 316
# Ref: Time Series, Pag. 317 - 362
# Ref: J Brownlee, 2017, Time Series Forecast Study 
# Ref: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# Ref: J Brownlee, 2020, Deep Learning for Time Series Forecasting, Editionv1.7 
# Ref: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Ref: https://www.statsmodels.org/stable/vector_ar.html#references




# -------------
# read datasets
# -------------
df_cal = pd.read_csv('calendar.csv')
df_sales_train = pd.read_csv('sales_train_validation.csv')
df_prices = pd.read_csv('sell_prices.csv')

df_cal.drop(columns =['month', 'year', 'event_type_1','event_name_2', 'event_type_2'])


#
# ---------------------
# Check for mising values
# ---------------------
# We can fill them in with a certain value (zero, mean/max/median by column, string) 
# --------------------
def values_missing(v):
    return sum(v.isnull())

print('Any missing value?:')
#df_sales_train.apply(values_missing, axis=0)   # testing missing values
#df_prices.apply(values_missing, axis=0)   # testing missing values
#df_cal.apply(values_missing, axis=0) 
#df_prices.apply(values_missing, axis=0) 

# clean columns with missing values NaN in the df_cal

df_cal = df_cal.fillna("")

# ------------------------------------------------------------------
# create a data frame traspose for data analysis and graphs analysis
# ------------------------------------------------------------------

# -------------------------------------------------
# Melt df sales_train because columns d1 ... d_1913
# -------------------------------------------------

df_sales_train_melted = pd.melt(df_sales_train, ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                var_name = 'd', value_name = 'sales')
# change values of columns d by a number
#df_sales_train_melted['d'] = df_sales_train_melted['d'].str.replace('d_', '')
df_sales_train_melted.head(2).T
df_sales_train_melted.tail(2).T

# ---------------------------------------
# join df_sales_train_melted to df_cal, add columns from df_cal (calendar)
# ---------------------------------------
# Merge df calendar with df sales_train_melted
# left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
# left_on: d,  Column to join on in the left DataFrame
# right_on: d, Column  to join on in the right DataFrame.

# change values column d in df_cal by a number
#df_cal['d'] = df_cal['d'].str.replace('d_', '')
#df_cal.head(3).T

# --------------------------------------
# Join df_sales_train_melted with df_cal
# create df for sales
# --------------------------------------
df_sales = df_sales_train_melted.merge(df_cal, left_on='d', 
                                                 right_on='d', how='left')
df_sales.head(2).T
df_sales.tail(2).T


# convert the 'date' column to datetime format 
df_sales['date']= pd.to_datetime(df_sales['date']) 

# ------------------------------------------------
# Descriptive columns 
# ------------------------------------------------
df_sales.columns
# Columns:
#     ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd',
#       'sales', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
#       'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
#       'snap_CA', 'snap_TX', 'snap_WI']


# Delete previous DF to save space and memory
del df_sales_train_melted

# ------------------------------------------------------
# obtain data in format arrays for ML Neural Networks algoritmics
# total sales per day for each category: food, hobbies, and household
# ------------------------------------------------------

fooddf=df_sales_train[df_sales_train['cat_id']=='FOODS']
hobbiesdf=df_sales_train[df_sales_train['cat_id']=='HOBBIES']
housedf=df_sales_train[df_sales_train['cat_id']=='HOUSEHOLD']

food=np.zeros((1,fooddf.shape[1]-6))
hobbies=np.zeros((1,hobbiesdf.shape[1]-6))
house=np.zeros((1,housedf.shape[1]-6))

fooddf
fooddf.shape

# ----------------------------------
# create arrays per day by categories
# -----------------------------------

for i in range(1,(fooddf.shape[1])-6):
    food[:,i]=sum(fooddf['d_{}'.format(i)])
    
for i in range(1,(fooddf.shape[1])-6):
    hobbies[:,i]=sum(hobbiesdf['d_{}'.format(i)])

for i in range(1,(fooddf.shape[1])-6):
    house[:,i]=sum(housedf['d_{}'.format(i)])

# --------------------------------------------
# Arrays data for ML algotimics Neural network
# --------------------------------------------
    
food   
hobbies
house

# Delete previous DF to save space and memory
del fooddf
del hobbiesdf
del housedf

# -------------------------------------------
# dataset sales ready
# -------------------------------------------

df_sales.head(2).T




# ------------------------------------------------
# Pre-procesing the data to ML algoitmics Time Series
# for category id: Foods, Hobbies, Household by date
# ------------------------------------------------

# ------------------------------------------
# Using technique: Split -> Apply -> Combine
# ------------------------------------------


# Analysis sales by category

df_cat_sales = df_sales.iloc[:, [3,7,8]] # only column cat_id, sales, and date

df_cat_sales = df_cat_sales.set_index('date')
df_cat_sales.dtypes
df_cat_sales.head()

# create df for each category and store id: split -> apply
df_cat_food_sales = df_cat_sales[df_cat_sales['cat_id'] == 'FOODS']
df_cat_food_sales = df_cat_food_sales.groupby(['date'])['sales'].sum() # sum of sales by date

df_cat_hobbies_sales = df_cat_sales[df_cat_sales['cat_id'] == 'HOBBIES']
df_cat_hobbies_sales = df_cat_hobbies_sales.groupby(['date'])['sales'].sum() # sum of sales by date

df_cat_household_sales = df_cat_sales[df_cat_sales['cat_id'] == 'HOUSEHOLD']
df_cat_household_sales = df_cat_household_sales.groupby(['date'])['sales'].sum() # sum of sales by date

df_cat_food_sales.head()
df_cat_food_sales.dtypes
df_cat_food_sales.index

# ------------------------------------------------
# Graphs descriptive: line plot
# ------------------------------------------------


plt.plot(df_cat_food_sales, color='blue')
plt.plot(df_cat_hobbies_sales, color='brown')
plt.plot(df_cat_household_sales, color='green')
plt.title('Daily sales by categories: Food, Hobbies, Household')
plt.show()

# -------------------------------------------
# Dealing with outliers
# -------------------------------------------
# Define threshold as minimum values
# the data present some values extreme that will be replaced by the mean point
# between the neighbours.

# ------------------------------------------------
# detecting outliers by graphs descriptive: line plot
# ------------------------------------------------
    
# According to the analysis of the graph, it will be used 5.000 
# as the threshold for products Food

df_cat_food_sales = df_cat_food_sales[df_cat_food_sales>5000]

    
# According to the analysis of the graph, it will be used 1.000 
# as the threshold for products Hobbies
    
df_cat_hobbies_sales = df_cat_hobbies_sales[df_cat_hobbies_sales>1000]

    
# According to the analysis of the graph, it will be used 2.500 
# as the threshold for products Household
df_cat_household_sales = df_cat_household_sales[df_cat_household_sales>2500]


plt.plot(df_cat_food_sales, color='blue')
plt.plot(df_cat_hobbies_sales, color='brown')
plt.plot(df_cat_household_sales, color='green')
plt.title('Daily sales by categories: Food, Hobbies, Household')
plt.show()

# -------
# combine
# -------
df_cat_sales = pd.concat([df_cat_food_sales, 
                  df_cat_hobbies_sales, 
                  df_cat_household_sales], 
                 axis=1, sort=False)

df_cat_sales.columns = ['Food', 'Hobbies', 'Household']


# -------------------------------------------------
# datasets sales for category ready for Ml Forecast
# -------------------------------------------------

df_cat_food_sales
df_cat_hobbies_sales
df_cat_household_sales

df_cat_sales # sales for the cat id Food, Hobbies, Household




# ----------------------------------
# read dataset submission
# Preparing the dataframe submission
# ----------------------------------
    
df_submissions = pd.read_csv('sample_submission.csv')

df_submissions.head(5)
df_submissions.tail(10)


df_submissions_predictions = df_submissions[df_submissions['id'].str.contains('(w*_validation)', regex=True)]
df_submissions_evaluation = df_submissions[df_submissions['id'].str.contains('(w*_evaluation)', regex=True)]


# --------------------------
# Validation data
# --------------------------
columns = df_submissions_predictions['id']
columns

df_submissions_predictions = df_submissions_predictions.T # traspose 

df_submissions_predictions.columns = columns # rename columns

df_submissions_predictions = df_submissions_predictions.iloc[1:] # delete columns id


# -------------------------
# Evaluation data
# -------------------------

columns = df_submissions_evaluation['id']
columns

df_submissions_evaluation = df_submissions_evaluation.T # traspose 

df_submissions_evaluation.columns = columns # rename columns

df_submissions_evaluation = df_submissions_evaluation.iloc[1:] # delete columns id

# ----------------------------------------------------
# Preparate df submissions for validation + evaluation
# ----------------------------------------------------

df_submissions_all = df_submissions.copy()

columns = df_submissions_all['id']
columns

df_submissions_all = df_submissions_all.T # traspose 

df_submissions_all.columns = columns # rename columns

df_submissions_all = df_submissions_all.iloc[1:] # delete columns id


# df_submissions ready


df_submissions_predictions # dataframe ready
df_submissions_evaluation


df_submissions_predictions.T 
df_submissions_evaluation.T
df_submissions_all.T


# -----------------------------------------------
# end Data Wrangling
# -----------------------------------------------







