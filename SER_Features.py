#!/usr/bin/env python
# coding: utf-8

# # DATA PREPARATION

# Importing Libraries 

# In[1]:


import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music, can be used to extract data from the audio files
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

# to play the audio files
from IPython.display import Audio

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Working with four different datasets ('RAVDESS', 'CREMA-D', 'TESS' and 'SAVEE'), created a dataframe to store all datasets.

# In[2]:


Ravdess = "audio_speech_actors_01-24/"
ravdess_directory_list = os.listdir(Ravdess)


# In[3]:


file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()


# In[4]:


Crema = "CremaD/AudioWAV/"
crema_directory_list = os.listdir(Crema)


# In[5]:


file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()


# In[6]:


Tess = "TESS/TESS Toronto emotional speech set data/"
tess_directory_list = os.listdir(Tess)


# In[7]:


file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()


# In[8]:


Savee = "SAVEE/ALL/"
savee_directory_list = os.listdir(Savee)


# In[9]:


file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()


# In[10]:


# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path.to_csv("data_path_all.csv",index=False)
data_path.head()


# We extracted few audio files of different emotions to check the samples

# In[11]:


emotion='fear'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
Audio(path)


# In[12]:


emotion='angry'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
Audio(path)


# In[13]:


emotion='sad'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
Audio(path)


# In[14]:


emotion='happy'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
Audio(path)


# Plot of the count of each emotions in the dataframe

# In[15]:


plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Use the `countplot` function from Seaborn
sns.countplot(x='Emotions', data=data_path)

# Add labels and title
plt.title('Count of Emotions', size=16)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)

# Remove unnecessary spines
sns.despine(top=True, right=True, left=False, bottom=False)

# Show the plot
plt.show()


# # DATA Augmentation

# Data augmentation is a technique used to artificially increase the size of a training dataset by applying 
# various transformations to the original data. The goal is to enhance the model's ability to generalize to different variations 
# of the input while maintaining the same label or output.
# 
# In the context of audio data, augmentation techniques such as 'Noise Injection, 'Shifting time', 'Changing pitch' and 'Changing speed' are employed

# In[16]:


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data,rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sample_rate, pitch_factor=0.7):
     return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)

# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)


# Here are the samples after data augmentation

# In[17]:


#sample audio
Audio(path)


# In[18]:


x = noise(data)
Audio(x, rate=sample_rate)


# In[19]:


x = stretch(data)
Audio(x, rate=sample_rate)


# In[20]:


x = shift(data)
Audio(x, rate=sample_rate)


# In[21]:


x = pitch(data, sample_rate)
Audio(x, rate=sample_rate)


# # Feature Extraction

# Feature extraction is a crucial step in the analysis audio data, as the datat isn't directly interpretable by models. Since models often require data in a more understandable format, feature extraction becomes essential. Feature extraction methods take raw audio input and produce feature vectors as output. This process involves transforming the original data into a set of relevant and informative features that can be used by models to uncover relationships and patterns within the data.

# Extracted follwoing features of the dataset:
# 'Zero Crossing Rate(ZCR)', 'Chroma Short-Time Fourier Transform (Chroma_stft)', 'Mel-Frequency Cepstral Coefficients (MFCCs)', 
# 'Root Mean Square (RMS) Value' and 'Mel Spectrogram'

# In[22]:


def extract_features(data, sample_rate):
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    return np.concatenate((zcr, chroma_stft, mfcc, rms, mel))

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio file
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # Extract features for the original data
    res1 = extract_features(data, sample_rate)

    # Data with noise
    noise_data = noise(data)  # Assuming noise function is defined elsewhere
    res2 = extract_features(noise_data, sample_rate)

    # Data with stretching and pitching
    new_data = stretch(data)  # Assuming stretch function is defined elsewhere
    data_stretch_pitch = pitch(new_data, sample_rate)  # Assuming pitch function is defined elsewhere
    res3 = extract_features(data_stretch_pitch, sample_rate)

    # Stack features vertically
    result = np.vstack((res1, res2, res3))

    return result


# Created training data (X and Y) by appending the features of corresponding emotions. This data is converted to pandas dataframe where data manipulation and modeling is performed later.

# In[23]:


X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for elements in feature:
        X.append(elements)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)


# In[24]:


len(X), len(Y), data_path.Path.shape


# In[25]:


Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features_SER.csv', index=False)
Features.head()


#  Plot of the count of each emotions after data augmentation and feature extraction.

# In[26]:


plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Use the `countplot` function from Seaborn
sns.countplot(x='labels', data=Features)

# Add labels and title
plt.title('Count of Emotions', size=16)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)

# Remove unnecessary spines
sns.despine(top=True, right=True, left=False, bottom=False)

# Show the plot
plt.show()


# In[27]:


print(Y)


# In[30]:


from collections import Counter
items = Counter(Y).keys()
print("No of unique items in the list are:", len(items))


# In[32]:


print(Counter(Y))


# Summary of dataset: Dataframe Features have 8 classes(multi-classification problem) and a class imbalance. 
# Data is exported to 'features.csv', which is imported in modeling notebook
