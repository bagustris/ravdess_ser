#!/usr/bin/evn python3
# python speech emotion revcognitin usin ravdess dataset

# import needed packages 
import glob  
import os  
import librosa  
import numpy as np  

from keras.utils import to_categorical
import ntpath

# function to extract feature
def extract_feature(file_name):   
    X, sample_rate = librosa.load(file_name)  
    stft = np.abs(librosa.stft(X))  
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T  
    mel = librosa.feature.melspectrogram(X, sr=sample_rate).T 
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T  
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T  
    return mfcc, chroma, mel, contrast, tonnetz

# function to parse audio file given main dir and sub dir
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features = [] #273: number of features, ori:193
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                print('process..', fn)
                feature  = extract_feature(fn)
            except Exception as e:
                print('cannot open', fn)
                continue
            #ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            ext_features = np.hstack(feature)
            features.append(ext_features)
            
    return np.array(features)

# change directory accordingly
main_dir = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/'
sub_dir = os.listdir(main_dir)  
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features = parse_audio_files(main_dir, sub_dir)  

# make sure dimension is OK
print(features.shape)

# If all is OK, let save it 
np.save('X_lld', features)
print("done") 

