#!/usr/bin/env python
# coding: utf-8

# # DATA MODELING

# Imported dataset 'features_SER', from data preparation file

# In[1]:


import pandas as pd
Features = pd.read_csv('features_SER.csv')
Features.head(1)


# In[2]:


X = Features.iloc[: ,:-1].values
Y = Features['labels'].values


# In[3]:


X.shape, Y.shape


# In[4]:


print(Y)


# In[5]:


import numpy as np
class_labels = np.unique(Y)

print(class_labels)


# As data has class imbalance, class weights are computed based on the distribution of classes within the dataset. These weights are applied during the model training process to prevent bias towards the majority classes.

# In[6]:


from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight

# Assuming y_train is your original class labels (not one-hot encoded)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=Y)

class_weight_dict = dict(enumerate(class_weights))

class_weights_actual = {i: class_weight_dict[i] for i in range(len(class_weight_dict))}

print(class_weights_actual)


# In[8]:


#Importing necessary libraries for model training

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np

num_classes = len(class_labels)

# Label encoding for class labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
num_classes = len(label_encoder.classes_)
y = to_categorical(Y_encoded, num_classes=num_classes)

# Splitting the data into training, validation, and test sets

x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


# In[9]:


print(y)


# In[11]:


# Scaling the data with sklearn's StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_val = scaler.transform(x_val)
X_test = scaler.transform(x_test)

X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# In[30]:


# Reshape data for LSTM input
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[31]:


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# In[32]:


input_shape_lstm = X_train_lstm.shape[1:]
lstm_model = build_lstm_model(input_shape_lstm)


# Two techniqies('Early Stopping' and 'ReduceLROnPlateau') of hyperparameter tuning are employed. These callbacks collectively contribute to effective training by preventing overfitting (EarlyStopping) and adjusting the learning rate to navigate more efficiently towards the optimal model configuration (ReduceLROnPlateau). They enhance the model's generalization ability and stability during the training process.

# In[12]:


early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


# In[34]:


# Train the model with class weights and callbacks
history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights_actual,
    callbacks=[early_stopping, reduce_lr]
)


# In[35]:


histroy_lstm = history


# In[36]:


# Make predictions on the test set
lstm_predictions = lstm_model.predict(np.array(X_test_lstm))


# In[37]:


print("Accuracy of LSTM model on test data : " , lstm_model.evaluate(x_test,y_test)[1]*100 , "%")


# In[38]:


epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
lstm_train_acc = histroy_lstm.history['accuracy']
lstm_train_loss = histroy_lstm.history['loss']
lstm_test_acc = histroy_lstm.history['val_accuracy']
lstm_test_loss = histroy_lstm.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , lstm_train_loss , label = 'Training Loss')
ax[0].plot(epochs , lstm_test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , lstm_train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , lstm_test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()


# In[46]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# In[45]:


y_pred = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
y_test_original = np.argmax(y_test, axis=1)


# In[47]:


# Plot the confusion matrix
class_labels = label_encoder.classes_
plot_confusion_matrix(y_test_original, y_pred, class_labels)


# In[48]:


print("Classification Report:")
print(classification_report(y_test_original, y_pred, target_names=class_labels))


# In[13]:


from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


# In[14]:


# Build the CNN model
def build_cnn_model(input_shape):
    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(162, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return model


# In[15]:


# Reshape data for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[16]:


input_shape_lstm = X_train_cnn.shape[1:]
cnn_model = build_cnn_model(input_shape_lstm)


# In[17]:


# Train the model with class weights and callbacks
history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    class_weight=class_weights_actual,
    callbacks=[early_stopping, reduce_lr]
)


# In[19]:


# Make predictions on the test set
lstm_predictions = cnn_model.predict(np.array(X_test_cnn))


# In[20]:


print("Accuracy of CNN model on test data : " , cnn_model.evaluate(X_test_cnn,y_test)[1]*100 , "%")


# In[21]:


cnn_history = history


# In[22]:


epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
cnn_train_acc = cnn_history.history['accuracy']
cnn_train_loss = cnn_history.history['loss']
cnn_test_acc = cnn_history.history['val_accuracy']
cnn_test_loss = cnn_history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , cnn_train_loss , label = 'Training Loss')
ax[0].plot(epochs , cnn_test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , cnn_train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , cnn_test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred_cnn, classes):
    cm = confusion_matrix(y_true, y_pred_cnn)
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# In[26]:


y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
y_test_original = np.argmax(y_test, axis=1)


# In[27]:


# Plot the confusion matrix
class_labels = label_encoder.classes_
plot_confusion_matrix(y_test_original, y_pred_cnn, class_labels)


# Summary: CNN outperforms LSTM on Speech Emotion Recognition task with the accuracy close to SOTA models on individual datasets
