#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.layers import Dropout  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt

# load feature data
X=np.load('X.npy')  
y=np.load('y.npy')  
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# DNN layer
n_dim = train_x.shape[1]  
n_classes = train_y.shape[1]  
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
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()  
# train the model  
history = model.fit(train_x, train_y, epochs=200, validation_data=[test_x, test_y], batch_size=4)

evaluate = model.evaluate(test_x, test_y, batch_size=4)
print(evaluate)

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
ax = axs[0]
ax.plot(history.history['loss'], label='train')
ax.plot(history.history['val_loss'], label='val')
#plt.legend()
ax.set_ylabel('loss')
ax.set_xlabel('epochs')
ax = axs[1] 
ax.plot(history.history['acc'], label='train')
ax.plot(history.history['val_acc'], label='val')
ax.set_ylabel('accuracy')
ax.set_xlabel('epochs')
plt.show()

# predicting emotion of audio test data from the model  
predict = model.predict(test_x,batch_size=4)

emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  

# predicted emotions from the test set  
y_pred = np.argmax(predict, 1)  
predicted_emo = []   
for i in range(0,test_y.shape[0]):  
    emo = emotions[y_pred[i]]  
    predicted_emo.append(emo)
    
actual_emo = []  
y_true = np.argmax(test_y, 1)  
for i in range(0,test_y.shape[0]):  
    emo = emotions[y_true[i]]  
    actual_emo.append(emo)
        
# generate the confusion matrix  
cm = confusion_matrix(actual_emo, predicted_emo)  
index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
cm_df = pd.DataFrame(cm, index, columns)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True)