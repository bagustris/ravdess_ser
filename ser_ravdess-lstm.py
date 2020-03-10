#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, GRU, Flatten, LSTM, Flatten 
from keras.layers import Dropout, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

# load feature data
X=np.load('X.npy')  
y=np.load('y.npy')
X = X.reshape((X.shape[0], 1, X.shape[1]))
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# DNN layer units
n_dim = train_x.shape[2]  
n_classes = train_y.shape[1]  

earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=75, restore_best_weights=True)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

# function to define model
def create_model():  
    model = Sequential()
    model.add(BatchNormalization(axis=-1, input_shape=(1, 193)))
    model.add(LSTM(n_dim, return_sequences=True, dropout=0.1, #input_shape=(1, 193),
                 recurrent_dropout=0.2))  
    model.add(LSTM(n_dim*2, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(n_dim, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(Flatten())
    #model.add(Dense(n_dim, activation='relu'))  
    #model.add(Dropout=0.4)
    model.add(Dense(n_classes, activation='softmax'))
              
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=50, batch_size=32, 
                 validation_data=[test_x[:150], test_y[:150]], callbacks=[earlystop])
print(max(hist.history['accuracy']), max(hist.history['val_accuracy']))
# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x[150:], test_y[150:], batch_size=32)
print(evaluate)
# [1.541983750131395, 0.6284722222222222]

# for plotting
#fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
#ax = axs[0]
#ax.plot(hist.history['loss'], label='train')
#ax.plot(hist.history['val_loss'], label='val')
#ax.legend()
#ax.set_ylabel('loss')
#ax.set_xlabel('epochs')
#ax = axs[1] 
#ax.plot(hist.history['acc'], label='train')
#ax.plot(hist.history['val_acc'], label='val')
#ax.legend()
#ax.set_ylabel('accuracy')
#ax.set_xlabel('epochs')
#plt.show()

## predicting emotion of audio test data from the model  
#predict = model.predict(test_x,batch_size=4)
#emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  

## predicted emotions from the test set  
#y_pred = np.argmax(predict, 1)  
#predicted_emo = []   
#for i in range(0,test_y.shape[0]):  
#    emo = emotions[y_pred[i]]  
#    predicted_emo.append(emo)
#    
#actual_emo = []  
#y_true = np.argmax(test_y, 1)  
#for i in range(0,test_y.shape[0]):  
#    emo = emotions[y_true[i]]  
#    actual_emo.append(emo)
#        
## generate the confusion matrix  
#cm = confusion_matrix(actual_emo, predicted_emo)  
#index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#cm_df = pd.DataFrame(cm, index, columns)                      
#plt.figure(figsize=(10,6))  
#sns.heatmap(cm_df, annot=True)
