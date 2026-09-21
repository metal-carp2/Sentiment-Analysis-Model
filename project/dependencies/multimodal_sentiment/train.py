from gc import callbacks
#from datasets import load_dataset, Audio
from dependencies.multimodal_sentiment.extra import extract_dataset, getFeatures
from dependencies.livedebug import show_live_debug_display
import random
import dependencies.parameters as param
from datasets import DownloadConfig
import numpy as np
from model_data import serialized_emotions
import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json  # Import the load_model function
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.python.keras.models import load_model
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
from ast import literal_eval
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from joblib import Parallel, delayed
import base64
import json

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

#Encode emotion labels into numbers


def decode_large_data(encoded_cell):
    decoded_json = base64.b64decode(encoded_cell.encode('utf-8')).decode('utf-8')
    return json.loads(decoded_json)  # Convert JSON string back to Python object

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)  # You can change `patience` as per requirement
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1)
#---------------------------
#---------------------------
def pad_2d_array(array, pad_value=0):
    # Find the maximum length in the second dimension
    max_length = max(len(row) for row in array)
    
    # Pad each row to match the maximum length
    padded_array = np.array([np.pad(row, (0, max_length - len(row)), constant_values=pad_value) for row in array])
    
    return padded_array

#model = load_model("./project/dependencies/multimodal_sentiment/model")
# Load the IEMOCAP Emotion Recognition dataset
model_version = "2.3"
dataset_version = "3.0"

model = load_model(f"./dependencies/multimodal_sentiment/models/v{model_version}")
model.compile(optimizer=Adam(0.0025), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
download_config = DownloadConfig(cache_dir="./cache_dir")

print("started reading")
dataset = pd.read_csv(
    f"./dependencies/multimodal_sentiment/datasets/v{dataset_version}.csv"
)

print("finished reading, starting converters")
# Convert columns after reading
dataset['mfcc'] = dataset['mfcc'].apply(decode_large_data)
dataset['text_embedding'] = dataset['text_embedding'].apply(decode_large_data)
dataset['label'] = dataset['label'].apply(decode_large_data)
print("finished converting")

def numpy1dfeatures(features):
    return np.array([np.array(i) for i in features])

def process_sublist(sublist, max_rows):
    """
    Processes a single sublist (2D jagged array).
    Pads its rows to `max_rows` and returns the padded result.
    """
    max_cols = len(sublist[0]) if sublist else 0
    padded_subarray = np.zeros((max_rows, max_cols), dtype=np.float64)
    for i, inner_list in enumerate(sublist):
        end_idx = min(len(inner_list), max_cols)
        padded_subarray[i, :end_idx] = inner_list[:end_idx]
    return padded_subarray

def pad2DFeatureTrainParallel(array):
    """
    Pads a 3D jagged list into a fixed-shape 3D NumPy array with parallelism.
    
    Parameters:
        array (list of list of lists): Jagged 3D list.
        
    Returns:
        np.ndarray: 3D NumPy array with consistent dimensions, padded with zeros.
    """
    # Single-pass shape calculation for axis 1 (rows)
    max_cols = 0
    max_rows = 0
    for sublist in array:
        max_rows = max(max_rows, len(sublist))  # Maximum number of rows
        for inner_list in sublist:
            max_cols = max(max_cols, len(inner_list))  # Maximum number of cols
    
    shape = (len(array), max_rows, max_cols)
    padded_array = np.zeros(shape, dtype=np.float64)
    
    # Parallel processing to process each sublist
    results = Parallel(n_jobs=-1)(  # Use parallelization for efficient processing
        delayed(process_sublist)(sublist, max_rows) for sublist in array
    )
    
    # Populate the shared padded_array
    for i, padded_subarray in enumerate(results):
        end_idx = min(len(padded_subarray), max_cols)
        padded_array[i, :, :] = padded_subarray[:, :end_idx]
    
    return padded_array

def standardize_3d_array_train(array):
    """
    Fits a StandardScaler on the training set and standardizes a 3D NumPy array.
    Standardization is applied across the flattened valid (non-zero) elements.

    Parameters:
    - array (np.ndarray): 3D NumPy array of shape (samples, rows, cols).

    Returns:
    - standardized_array (np.ndarray): Standardized 3D NumPy array.
    - scaler (StandardScaler): Fitted StandardScaler for reuse on test data.
    """
    array = np.asarray(array)
    flattened_data = np.concatenate([
        sample[sample != 0].reshape(-1)  # Flatten non-padded values
        for sample in array
    ])
    flattened_data = flattened_data.reshape(-1, 1)  # Reshape for StandardScaler
    
    # Fit the StandardScaler on training data
    scaler = StandardScaler()
    scaler.fit(flattened_data)

    # Standardize training data
    standardized_array = np.zeros_like(array, dtype=np.float32)
    for i, sample in enumerate(array):
        mask = sample != 0  # Identify non-padded values
        sample_values = sample[mask].reshape(-1, 1)
        if sample_values.size > 0:
            standardized_values = scaler.transform(sample_values).flatten()
            sample[mask] = standardized_values
        standardized_array[i] = sample

    return standardized_array, scaler

def standardize_3d_array_test(array, scaler):
    """
    Applies a pre-fitted StandardScaler to a test set to standardize a 3D NumPy array.

    Parameters:
    - array (np.ndarray): 3D NumPy array of shape (samples, rows, cols).
    - scaler (StandardScaler): Pre-fitted StandardScaler from the training set.

    Returns:
    - standardized_array (np.ndarray): Standardized 3D NumPy array.
    """
    array = np.asarray(array)
    standardized_array = np.zeros_like(array, dtype=np.float32)
    
    for i, sample in enumerate(array):
        mask = sample != 0  # Identify non-padded values
        sample_values = sample[mask].reshape(-1, 1)
        if sample_values.size > 0:
            standardized_values = scaler.transform(sample_values).flatten()
            sample[mask] = standardized_values
        standardized_array[i] = sample

    return standardized_array

# Function to create and return a fitted scaler
def create_scaler(train_array, padding_value=0.0):
    """
    Creates a StandardScaler and fits it to the valid (non-padded) values
    of the train array.
    """
    scaler = StandardScaler()
    valid_values = train_array[train_array != padding_value].reshape(-1, 1)
    scaler.fit(valid_values)
    return scaler

# Function to normalize padded arrays using the given scaler
def normalize_padded_array(array, scaler, padding_value=0.0):
    """
    Normalizes a padded array while leaving padding unchanged.
    """
    normalized_array = np.zeros_like(array)  # Create an array of same shape
    for i in range(array.shape[0]):  # Iterate over each sample
        valid_mask = array[i] != padding_value  # Mask for valid (non-padded) values
        valid_data = array[i][valid_mask]  # Extract valid data
        if len(valid_data) > 0:  # Skip empty rows
            normalized_valid_data = scaler.transform(valid_data.reshape(-1, 1)).flatten()
            normalized_array[i][valid_mask] = normalized_valid_data  # Assign normalized values
    return normalized_array

dataset = dataset[dataset["major_emotion"] != "other"]
dataset = dataset[dataset["major_emotion"] != "disgust"]

X=dataset[["mfcc"]]

y=dataset["major_emotion"]

print(y.value_counts())
while True:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
    if len(np.unique(y_test)) == 8 and len(np.unique(y_train)) ==8:
        break

print(np.unique(y_train))
#mfcc_train = pad2DFeatureTrainParallel(X_train["mfcc"])
mfcc_train = pad_sequences(X_train["mfcc"])
#text_train = numpy2dfeatures(X_train["text_embedding"], True)
label_train = to_categorical(lb.fit_transform(y_train))

print(mfcc_train.shape)
#mfcc_test = pad2DFeatureTrainParallel(X_test["mfcc"])
mfcc_test = pad_sequences(X_test["mfcc"])
#text_test = numpy2dfeatures(X_test["text_embedding"], True)
label_test = to_categorical(lb.fit_transform(y_test))

print(label_train.shape)
print(label_test.shape)
scaler = create_scaler(mfcc_train, 0.0)
mfcc_train = normalize_padded_array(mfcc_train, scaler, 0.0)
mfcc_test = normalize_padded_array(mfcc_test, scaler, 0.0)

'''
scaler = StandardScaler()
scaler.fit(mfcc_train)
mfcc_train = scaler.transform(mfcc_train)
mfcc_test = scaler.transform(mfcc_test)
'''
'''
mfcc_train, scaler = standardize_3d_array_train(mfcc_train)
mfcc_test = standardize_3d_array_test(mfcc_test, scaler)
print(mfcc_train.shape)
'''
mfcc_train = np.expand_dims(mfcc_train, axis=-1)
mfcc_test = np.expand_dims(mfcc_test, axis=-1)

history = model.fit(
    x=[
        mfcc_train
    ],
    y=label_train,
    batch_size = 256,
    epochs = 40,
    verbose = 1,
    validation_data = ([mfcc_test], label_test),
    callbacks=[lr_scheduler]
)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# (40, n_timestep)