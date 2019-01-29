# Recurrent Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing local functions
import stock_raw_data

#creating dataframe
stock_name_local = 'adani_power'
stock_data = stock_raw_data.stock_data(stock_name_local).stock_data_func()
stock_data = stock_data[['new_date','Open Price']]
dataset_train = stock_data.iloc[:2300]
# print(dataset_train.shape())
training_set = dataset_train.iloc[:, 1:2].values
# print(training_set)

### Use normalisation whenever we are using RNN or LSTM

# # Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# print(len(training_set_scaled))
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 2300):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# print(X_train,y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# print(X_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 20, batch_size = 40)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
# dataset_test = quandl.get("NSE/idea", start_date='2019-01-01', end_date='2019-01-10')
dataset_test = stock_data.iloc[2301:]
# for i in [36.297,36.1,36.8,45.78,36.78,26.85,78.26,14.25]:
#     dataset_test = dataset_test.append([{'Open Price':i}],ignore_index=True)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open Price'], dataset_test['Open Price']), axis = 0)

inputs_raw = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
print(type(inputs_raw))
# print(inputs)
inputs = inputs_raw.reshape(-1,1)
inputs = sc.transform(inputs)
print(inputs)
# X_test = []
# for i in range(60, 101):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted_stock_price = regressor.predict(X_test)
# print(predicted_stock_price)
j=0
while j < 10 :
    X_test=[]
    for i in range(60,j+101):
        # inputs.append(temp)
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_test)
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print(predicted_stock_price[0][-1])
    inputs_raw = np.concatenate((inputs_raw, np.array([predicted_stock_price[0][-1]])))
    inputs = inputs_raw.reshape(-1, 1)
    inputs = sc.transform(inputs)
    print(len(inputs))
    # temp = predicted_stock_price[0][-1]
    # inputs = np.concatenate((inputs, np.array([[temp]])))
    j += 1



print(predicted_stock_price)
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = stock_name_local)
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted {0} Stock Price'.format(stock_name_local))
plt.title('{0} Stock Price Prediction'.format(stock_name_local))
plt.xlabel('Time')
plt.ylabel('{0} Stock Price'.format(stock_name_local))
plt.legend()
plt.show()
