#!/usr/bin/evn python3
# python speech emotion revcognitin usin ravdess dataset

# import needed packages 
import glob  
import os  
import librosa  
import numpy as np  

from keras.utils import to_categorical
import ntpath

# number of window
lenmin = 100

# function to extract feature
def extract_feature(file_name): 
    X, sample_rate = librosa.load(file_name)  
    stft = np.abs(librosa.stft(X))  
    mfcc_i = librosa.feature.mfcc(X, sample_rate, n_mfcc=40)
    if mfcc_i.shape[1] < lenmin:
        mfcc_i = np.hstack((mfcc_i, np.zeros((mfcc_i.shape[0], lenmin-mfcc_i.shape[1]))))
    mfcc = mfcc_i[:,:lenmin]
    #mfcc = np.mean(mfccs.T, axis=0)   # 40 features, mean mfcc
    mfccd = librosa.feature.delta(mfcc)
    mfccdd= librosa.feature.delta(mfcc, order=2)
    return mfcc, mfccd, mfccdd

# function to parse audio file given main dir and sub dir
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 120)), np.empty(0)  #273: number of features, ori:193
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                print('process..', fn)
                mfcc, mfcc_delta, mfcc_delta2  = extract_feature(fn)
            except Exception as e:
                print('cannot open', fn)
                continue
            #ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            ext_features = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta2.T])
            features = np.append(features, ext_features)
            filename = ntpath.basename(fn)
            labels = np.append(labels, filename.split('-')[2])  # grab 3rd item
            #labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int) # 

# change directory accordingly
main_dir = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/'
sub_dir = os.listdir(main_dir)  
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features, labels = parse_audio_files(main_dir, sub_dir)  
labels_oh = to_categorical(labels)   # one hot conversion from integer to binary
print("done") 

# make sure dimension is OK
print(features.shape)
print(labels.shape)

# remove first column because label start from 1 (not from 0)
#labels_oh = labels_oh[:,1:]

# If all is OK, let save it 
np.save('X_mfcc', features)
#np.save('y', labels_oh)