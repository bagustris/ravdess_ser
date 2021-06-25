#!/usr/bin/evn python3
# python speech emotion revcognitin usin ravdess dataset

# import needed packages 
import os  
import librosa  
import numpy as np  

from keras.utils import to_categorical
import ntpath

import sys
sys.path.append('/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/')
from read_csv import load_features

# number of window
lenmin = 523    #longest sequence

# function to parse audio file given csv file
def parse_audio_files(parent_dir):
    features = [] 
    labels = np.empty(0)  #273: number of features, ori:193
    for fn in os.listdir(parent_dir):
        print('process...', fn)
        feature_i = load_features(parent_dir+fn, delim=';')
        if feature_i.shape[0] < lenmin:
            feature_i = np.vstack([feature_i, np.zeros((lenmin-feature_i.shape[0], feature_i.shape[1]))])
        feature_i = feature_i[:,:lenmin]
        features.append(feature_i)
        filename = ntpath.basename(fn)
        labels = np.append(labels, filename.split('-')[2])  # grab 3rd item
    return np.array(features), np.array(labels, dtype = np.int) # 

# change directory accordingly
main_dir = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/audio_features_egemaps/'
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features, labels = parse_audio_files(main_dir)  
labels_oh = to_categorical(labels)   # one hot conversion from integer to binary
print("done") 

# make sure dimension is OK
print(features.shape)
print(labels.shape)

# remove first column because label start from 1 (not from 0)
labels_oh = labels_oh[:,1:]

# If all is OK, let save it 
np.save('X_egemaps', features)
np.save('y_egemaps', labels_oh)
