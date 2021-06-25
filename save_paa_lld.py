#!/usr/bin/evn python3
# python speech emotion revcognitin usin ravdess dataset

# import needed packages 
import glob  
import os  
import librosa  
import numpy as np  

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

import sys, traceback

# function to extract feature
def extract_feature(file_name):   
    [Fs, x] = audioBasicIO.read_audio_file(file_name)
    if x.ndim == 2:
        x = x[:,0]  
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.025*Fs, 0.010*Fs)
    return F.T

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
                traceback.print_exc()
                sys.exit(3)
            features.append(feature)
            
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
np.save('X_paa', features)
print("done") 

