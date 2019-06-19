#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.layers import Dropout  
import matplotlib.pyplot as plt

# load feature data
X=np.load('X.npy')  
y=np.load('y.npy')  

# DNN layer
n_dim = X.shape[1]  
n_classes = y.shape[1]  
n_hidden_units_1 = n_dim  
n_hidden_units_2 = 400 # approx n_dim * 2  
n_hidden_units_3 = 200 # half of layer 2  
n_hidden_units_4 = 100

# function to define model
def create_model(activation_function='relu', init_type='normal', optimiser='adam', dropout_rate=0.2):  
    model = Sequential()  
    # layer 1  
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, init=init_type, activation=activation_function))  
    # layer 2  
    model.add(Dense(n_hidden_units_2, init=init_type, activation=activation_function))  
    model.add(Dropout(dropout_rate))  
    # layer 3  
    model.add(Dense(n_hidden_units_3, init=init_type, activation=activation_function))  
    model.add(Dropout(dropout_rate))  
    #layer4  
    model.add(Dense(n_hidden_units_4, init=init_type, activation=activation_function))  
    model.add(Dropout(dropout_rate))  
    # output layer  
    model.add(Dense(n_classes, init=init_type, activation='softmax'))  
    #model compilation  
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()  
# train the model, because the model is distributed based on category
# using validation split is not good
history = model.fit(X, y, epochs=200, batch_size=4, validation_split=0.3)

acc = history.history['val_acc']
print(max(acc))
