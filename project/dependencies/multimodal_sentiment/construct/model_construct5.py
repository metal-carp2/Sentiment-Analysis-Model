from ast import Global
from keras import Model, Input, Sequential
from keras.layers import (
    Dense, Conv1D, LSTM, Bidirectional, Dropout, Flatten,
    GlobalAveragePooling1D, Attention, Concatenate, LayerNormalization,
    MultiHeadAttention, Layer, BatchNormalization, MaxPooling1D, MaxPooling2D, Masking
)
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K

#Build sequential CNN
CNN_model = Sequential()

#Build first layer
CNN_model.add(Conv1D(16, 5,padding='same',
                 input_shape=(40, 1), activation='relu'))
CNN_model.add(Dropout(0.2))
#Build second layer
CNN_model.add(Conv1D(32, 5,padding='same',activation='relu'))
CNN_model.add(Dropout(0.5))
#Build third layer
CNN_model.add(Conv1D(64, 5,padding='same',activation='relu'))
CNN_model.add(Dropout(0.4))
#Build forth layer
CNN_model.add(Conv1D(128, 5,padding='same',activation='relu'))
CNN_model.add(BatchNormalization())
#Add dropout
CNN_model.add(Dropout(0.4))

#Flatten 
CNN_model.add(Flatten())

CNN_model.add(Dense(128, activation ='relu'))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(64, activation ='relu'))
CNN_model.add(Dense(9, activation='softmax'))

# Compile the model
CNN_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Model summary
CNN_model.summary()

# Save the model
version = "2.1.3"
CNN_model.save(f"./project/dependencies/multimodal_sentiment/models/v{version}")

