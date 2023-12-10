# Speech-Emotion-Recognition

This project aims to develop a deep learning model capable of recognizing emotions in spoken language. In the constant exchange of speech, the ability to discern emotions becomes crucial for a more personalized and empathetic interaction.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [File Structure](#FileStructure)
- [Conclusion](#Conclusion)

## Getting Started
Data sets and softwares to be installed are provided below to set up and deploy the emotion recognition system on your local machine.

### Prerequisites
Download datasets from below links:  


RAVDESS  -- https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

SAVEE -- https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee 

TESS -- https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

CREMA D -- https://www.kaggle.com/datasets/ejlok1/cremad


### Installation
You will need the following to run files:

Python  


Pandas  


Librosa   


numpy   


Seaborn  


matplotlib

## File Structure
SER_Features.py : Contains preprocessing of the data and extracting all features in a csv file.  

SER_Modeling.py : Contains SER models with LSTM and CNN .  

## Conclusion
CNN outperforms LSTM on Speech Emotion Recognition task with the accuracy close to SOTA models on individual datasets


