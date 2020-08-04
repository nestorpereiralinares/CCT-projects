# --------------------------------------------------
# Multivariate Multi-step encoder-decoder LSTM model
# --------------------------------------------------
#  Prediction 3 features 30 time step
# ------------------------------------------------
# Multivariate LSTM Models: Multiple Parallel Series

# Ref: J. Brownlee, 2020, Deep Learning for Time Series Forecasting, Edition: v1.7
# Ref: G. Bontempi, S. Ben, Y. Le, 2013, Machine Learning Strategies for Time Series Forecasting, ResearchGate.net
#    https://www.researchgate.net/publication/236941795_Machine_Learning_Strategies_for_Time_Series_Forecasting

import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# Split a multivariate sequence into samples
# ref: J. Brownlee, 2020, Deep Learning for Time Series Forecasting, Edition: v1.7
#
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# -----------------------------------------
# data preparation and modeling for multiple parallel input (multivariate)
# and multi-step output
# ref: J. Brownlee, 2020, Deep Learning for Time Series Forecasting, Edition: v1.7
# ---------------------------------------


# define input sequence
food.T
hobbies.T
house.T
dataset = hstack((food.T, hobbies.T, house.T))
dataset.shape
dataset

# choose a number of time steps
# It uses 10 prior time steps of each three(3) features or input times series
# for food, hobbies and household to predict 30 time steps of the 3 features.
# -----------------------------
n_steps_in = 10
n_steps_out = 30
n_features = 3
# ----------------------------
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features: 3

X.shape
y.shape

# ---------------------------------
nobs = 30 # predictions for 30 days
futuredays = 28 # final prediction
# ---------------------------------

X_futuredays = X[-futuredays:] # data for 28 days final prediction

X_train = X[0:-(nobs+futuredays)]  #  data for training
X_train
X_test = X[-(nobs+futuredays):]  # data for 30 days of test prediction 
X_test

Y_train = y[0:-(nobs+futuredays)]  #  data for training
Y_train
Y_test = y[-(nobs+futuredays):]  # data for 30 days of test prediction
Y_test


samples for for three(30 features )

# --------------------------
# Define model
# Long Short-Term Memory networks, or LSTMs for short, can be applied to time series forecasting, 
# making predictions based on time series data.
# Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture.
# ref: J. Brownlee, 2020, Deep Learning for Time Series Forecasting, Edition: v1.7,Pag. 123
# --------------------------
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features))) # encoder models several features
model.add(RepeatVector(n_steps_out))   # encoder is repeated for each time step required                                       
model.add(LSTM(200, activation='relu', return_sequences=True)) # decoder single output for each features
model.add(TimeDistributed(Dense(n_features))) # make each time step fron the decoder
model.compile(optimizer='adam', loss='mse')




# fit model
model.fit(X_train, Y_train, epochs=300, verbose=0)



# ---------------------------------
nobs = 30 # predictions for 30 days
# ---------------------------------

X_test  # data for 30 days of prediction
Y_test

Y_prediction = Y_test


Y_prediction = model.predict(X_test, verbose=0)
# Convert to float64
Y_prediction = Y_prediction.astype(np.float64)

print(Y_prediction)

Y_test.shape[2]
Y_prediction.shape


# --------------------
# Compare first values
# --------------------
p = ['Food', 'Hobbies', 'Household']

for i in range(Y_test.shape[2]):
        j=0
        print('Product: ', p[i])
        print(' Data for test: ')
        print(Y_test[:,j,i])
        print('\n Data predicted: ')
        print(Y_prediction[:,j,i])
    
# ----------------------
# Evaluation performance by graphs 
# ----------------------   

for i in range(Y_test.shape[2]):
    for j in range(0,30,30):
        plt.figure(figsize=(10,10))
        plt.title("30 days ahead prediction on the product: %s" 
                  % (p[i]))
        plt.plot(Y_test[:,j,i].ravel())
        plt.plot(Y_prediction[:,j,i].ravel())
    

# ----------------------
# Evaluate performance
# ----------------------
# Evaluate RMSE

# ---------------------------------
# Calculate and compare MSE 
# ---------------------------------

# Compute the mean square error

p = ['Food', 'Hobbies', 'Household']

for i in range(Y_test.shape[2]):
    for j in range(0,30,30):
        mse = ((Y_prediction[:,j,i] - Y_test[:,j,i]) ** 2).mean()
        rmse = np.sqrt(mse)
        print('Product : ',p[i])
        print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
        print('The Root Mean Square Error (RMSE) of the forecast: {:.4f}'.format(round(rmse, 2)))




# ---------------------------------
# futuredays = 28 final prediction
# ---------------------------------     

X_futuredays

Y_prediction = model.predict(X_futuredays, verbose=0)
# Convert to float64
Y_prediction = Y_prediction.astype(np.float64)

print(Y_prediction)

Y_test.shape[2]
Y_prediction.shape

# --------------------------------
# visualize prediction for 28 days
# --------------------------------
p = ['Food', 'Hobbies', 'Household']


for i in range(Y_prediction.shape[2]):
    for j in range(0,30,30):
        plt.figure(figsize=(10,10))
        plt.title("30 days ahead prediction on the product: %s" 
                  % (p[i]))
        #plt.plot(Y_test[:,j,i].ravel())
        plt.plot(Y_prediction[:,j,i].ravel(), color='g')
    


