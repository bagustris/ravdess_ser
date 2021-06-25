#!/usr/bin/env python3 

# load needed modules
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Flatten, Dropout, GRU , BatchNormalization
 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load feature data
X = np.load('X_egemaps.npy')
y = np.load('y_egemaps.npy')  
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# DNN layer units
n_dim = train_x.shape[2]  
n_classes = train_y.shape[1]  

## normalize data
mean = train_x.reshape(504172, 23).mean(axis=0)
train_x -= mean
std = train_x.reshape(504172, 23).std(axis=0)
train_x /= std

test_x -= mean
test_x /= std

# function to define model
def create_model(activation_function='relu', init_type='normal', optimiser='rmsprop'):  
    model = Sequential()  
    # layer 1  
    #model.add(BatchNormalization(axis=-1, input_shape=(523, 23)))
    model.add(GRU(n_dim, activation='relu', input_shape=(523, 23),
                  dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(100, activation='relu', dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, kernel_initializer=init_type, activation='softmax'))  
    # model compilation 
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])  
    return model
   
# create the model  
model = create_model()  
print(model.summary())

# train the model  
hist = model.fit(train_x, train_y, epochs=100, validation_data=[test_x, test_y], batch_size=32)

# evaluate model, test data may differ from validation data
evaluate = model.evaluate(test_x, test_y, batch_size=4)
print(evaluate)    

# for plotting
fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
ax = axs[0]
ax.plot(hist.history['loss'], label='train')
ax.plot(hist.history['val_loss'], label='val')
ax.legend()
ax.set_ylabel('loss')
ax.set_xlabel('epochs')
ax = axs[1] 
ax.plot(hist.history['acc'], label='train')
ax.plot(hist.history['val_acc'], label='val')
ax.legend()
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
columns = index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
#columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  
cm_df = pd.DataFrame(cm, index, columns)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True)