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


# Build Sequential CNN
CNN_model = Sequential()

# Add Masking layer to ignore padded values
CNN_model.add(Masking(mask_value=0.0, input_shape=(None, 40)))  # Shape: (features, time_steps)
# First Conv1D layer
CNN_model.add(Conv1D(8, 6, strides=4, padding='valid', activation='relu'))
CNN_model.add(MaxPooling1D(pool_size=2, padding="valid"))
CNN_model.add(Dropout(0.2))
# Second Conv1D layer
CNN_model.add(Conv1D(16, 4, padding='same', activation='relu'))
CNN_model.add(MaxPooling1D(pool_size=2, padding="valid"))
CNN_model.add(Dropout(0.2))
# Third Conv1D layer
CNN_model.add(Conv1D(32, 2, padding='same', activation='relu'))
CNN_model.add(Conv1D(32, 2, padding='same', activation='relu'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Conv1D(16, 2, padding='same', activation='relu'))
CNN_model.add(MaxPooling1D(pool_size=2, padding="valid"))
CNN_model.add(Dropout(0.2))


# Add Dense layers for classification
CNN_model.add(Dense(24, activation='relu'))
CNN_model.add(GlobalAveragePooling1D())  # Aggregates the sequence dimension (axis 1)
CNN_model.add(Dropout(0.2))
CNN_model.add(Dense(16, activation='relu'))
CNN_model.add(Dense(9, activation='softmax'))  # For 9 classes

# Compile the model
CNN_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Model summary
CNN_model.summary()

# Save the model
version = "2.1.2"
CNN_model.save(f"./project/dependencies/multimodal_sentiment/models/v{version}")