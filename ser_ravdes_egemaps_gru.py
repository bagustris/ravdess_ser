#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, Dropout, GRU , BatchNormalization
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load feature data
X = np.load('X_egemaps.npy')
y = np.load('y_egemaps.npy')  


# z-score normalization
ori_aud_features = X
norm_aud_features = []
for aud_original in ori_aud_features:
    aud_original_np = np.asarray(aud_original)
    z_norm_aud = (aud_original_np - aud_original_np.mean()) / aud_original_np.std()
    norm_aud_features.append(np.around(z_norm_aud, 6))

X = np.array(norm_aud_features)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# nodes
n_dim = train_x.shape[2]

# DNN layer units
n_dim = np.array(train_x).shape[2]  
n_classes = np.array(train_y).shape[1]  

## normalize data
#mean = train_x.reshape(504172, 23).mean(axis=0)
#train_x -= mean
#std = train_x.reshape(504172, 23).std(axis=0)
#train_x /= std

#test_x -= mean
#test_x /= std

# define earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

## function to define model
def gru_model():  
    model = Sequential()  
    # layer 1  
    model.add(BatchNormalization(axis=-1, input_shape=(523, 23)))
    model.add(GRU(n_dim, return_sequences=True, dropout=0.1, #input_shape=(1, 193),
                 recurrent_dropout=0.2))  
    model.add(GRU(n_dim*4, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(n_dim*4, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(n_dim*4, dropout=0.1, recurrent_dropout=0.2))  
    model.add(Dense(n_dim, activation='relu'))  
    #model.add(Dropout=0.4)
    model.add(Dense(n_classes, activation='softmax'))
              
    # model compilation  
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    return model

# create the model  
model = gru_model()  
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=100, validation_data=[test_x, test_y], 
                 batch_size=32, callbacks=[earlystop])

# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x, test_y, batch_size=32)
print(evaluate)    

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
#columns = index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
##columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#cm_df = pd.DataFrame(cm, index, columns)                      
#plt.figure(figsize=(10,6))  
#sns.heatmap(cm_df, annot=True)
