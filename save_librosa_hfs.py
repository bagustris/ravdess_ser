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
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)  
    mfcc_mean = np.mean(mfccs.T, axis=0)   # 40 features, mean mfcc
    mfcc_std = np.std(mfccs.T, axis=0)
    chroma_mean = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)  
    chroma_std = np.std(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)  
    mel_mean = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)  
    mel_std = np.std(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)  
    contrast_mean = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    contrast_std = np.std(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    tonnetz_std = np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)  
    return (mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean,
           mfcc_std, chroma_std, mel_std, contrast_std, tonnetz_std)
           
# function to parse audio file given main dir and sub dir
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features = []  #273: number of features, ori:193
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                print('process..', fn)
                features_i  = extract_feature(fn)
            except Exception as e:
                print('cannot open', fn)
                continue
            #ext_features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            ext_features = np.hstack(features_i)
            features.append(ext_features)
            #filename = ntpath.basename(fn)
            #labels = np.append(labels, filename.split('-')[2])  # grab 3rd item
            #labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features) #, np.array(labels, dtype = np.int) # 

# change directory accordingly
main_dir = '/media/bagustris/bagus/dataset/Audio_Speech_Actors_01-24/'
sub_dir = os.listdir(main_dir)  
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features = parse_audio_files(main_dir, sub_dir)  
#labels_oh = to_categorical(labels)   # one hot conversion from integer to binary


# make sure dimension is OK
print(features.shape)
#print(labels.shape)

# remove first column because label start from 1 (not from 0)
#labels_oh = labels_oh[:,1:]

# If all is OK, let save it 
np.save('X_meanstd', features)
#np.save('y_dd', labels_oh)
print("done")   
