# keras demo timeditributed
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM

# prepare sequence
length = 5
seq = np. array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)

# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000

# create LSTM without timedistributed: many-to-one
def build_model():
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(length, 1)))
    model.add(Dense(length))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

without_td = build_model()
print(without_td.summary())

# train LSTM
without_td.fit(X, y.reshape(5), epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result_wtd = without_td.predict(X, batch_size=n_batch, verbose=0)
for value in result_wtd[0,:]:
	print('%.1f' % value)

    
# create LSTM with timedistibuted: many-to-many
def build_tdmodel():
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

model = build_tdmodel()
print(model.summary())

# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)

# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)

print('Output: ')
for value in result[0,:,0]:
	print('%.1f' % value)