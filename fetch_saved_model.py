from keras.models import load_model

regressor = load_model('e55_b32_optadam.h5')

print(regressor.summary())


# Recurrent Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing local functions
import stock_raw_data_source

#creating dataframe
stock_name_local = 'adani_power'
stock_data = stock_raw_data_source.stock_data(stock_name_local).stock_data_func()
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
# for i in range(60, 2300):
#     X_train.append(training_set_scaled[i-60:i, 0])
#     y_train.append(training_set_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)

# print(X_train,y_train)
# Reshaping
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
# dataset_test = quandl.get("NSE/idea", start_date='2019-01-01', end_date='2019-01-10')
dataset_test = stock_data.iloc[2301:]
for i in [51.297,51.1,51.8,51.20,51.78,51.85,51.26,20.25]:
    dataset_test = dataset_test.append([{'Open Price':i}],ignore_index=True)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open Price'], dataset_test['Open Price']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 109):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = stock_name_local)
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted {0} Stock Price'.format(stock_name_local))
plt.title('{0} Stock Price Prediction'.format(stock_name_local))
plt.xlabel('Time')
plt.ylabel('{0} Stock Price'.format(stock_name_local))
plt.legend()
plt.show()
