import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Concatenate, Dropout, LSTM, Reshape, Masking, Flatten, GlobalAveragePooling1D, Attention


from sklearn.metrics import accuracy_score

# Define Inputs
pitch_input = Masking(mask_value=0.0)(Input(shape=(None,), name="pitch_data"))  # shape: (n,)
volume_input = Masking(mask_value=0.0)(Input(shape=(None,), name="volume_data"))  # shape: (n,)
mfcc_input = Masking(mask_value=0.0)(Input(shape=(None,), name="mfcc_data"))  # shape: (n,)
dialogue_embedding_input = Input(shape=(384,), name="dialogue_embedding")  # shape: (384,)

# Reshape the inputs to match the LSTM requirements (each time step has one feature)
dialogue_reshaped = Reshape((-1, 1), name="reshape_dialogue")(dialogue_embedding_input)
pitch_reshaped = Reshape((-1, 1), name="reshape_pitch")(pitch_input)  # (n, 1)
volume_reshaped = Reshape((-1, 1), name="reshape_volume")(volume_input)  # (n, 1)
mfcc_reshaped = Reshape((-1, 1), name="reshape_mfcc")(mfcc_input)  # (n, 1) reshape MFCC correctly

# Process sequential data with LSTMs
pitch_lstm = LSTM(32, activation='relu', name='pitch_lstm')(pitch_reshaped)
volume_lstm = LSTM(32, activation='tanh', name='volume_lstm')(volume_reshaped)
mfcc_lstm = LSTM(32, activation='relu', name='mfcc_lstm')(mfcc_reshaped)

# Concatenate Pitch, Volume, and MFCC features
pm_concat = Concatenate(name='concatenate_pitch_mfcc')([pitch_lstm, mfcc_lstm])
pitch_bias = Dense(32, activation='tanh', name='pitch_bias_dense')(pm_concat)

# Concatenate Pitch + MFCC Dense and Volume LSTM features
concat_vpm = Concatenate(name='concatenate_pitch_volume')([pitch_bias, volume_lstm])
total_conv = Dense(64, activation='relu', name='total_dense')(concat_vpm)

# Fully connected layers for feature integration
dense1 = Dense(128, activation='relu', name='dense1')(dialogue_reshaped)
dropout1 = Dropout(0.3, name='dropout1')(dense1)
dense2 = Dense(64, activation='tanh', name='dense2')(dropout1)  # Reduced the number of units to 32
# Print shapes of flattened tensotrs before concatenatio

pooled = GlobalAveragePooling1D()(dense2)  # Shape: (None, 64)
# Combine all processed features (now both reshaped/flattened to the same dimension)
combined = Concatenate(name='concatenate_features')([total_conv, pooled])

# Print the shape of the concatenated tensor before passing to the final dense layer
print(f"Shape of combined features: {combined.shape}")

# Final dense layer after combining features
final_dense = Dense(64, activation='relu', name='final_dense')(combined)

# Output layer for multi-label regression (one node per emotion percentage prediction)
output = Dense(9, activation='sigmoid', name='output')(final_dense)  # Assuming 9 emotion labels to predict percentages (0-1 scale)

# Define the model
model = Model(
    inputs=[pitch_input, volume_input, mfcc_input, dialogue_embedding_input],
    outputs=output
)

# Compile the model for multi-label regression
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',  # Correct for multi-label classification with sigmoid
    metrics=['accuracy']
)

# Display the model summary
model.summary()

import os
print("Current Working Directory:", os.getcwd())

# Save the model
model.save("./project/dependencies/multimodal_sentiment/model")