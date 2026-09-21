from keras.models import Sequential
from keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout, Bidirectional, LSTM,
    Dense, Flatten, Masking
)
from keras.optimizers import Adam

# Model definition
model = Sequential()
model.add(Masking(0.0, input_shape=(None, 1)))
# Convolutional layers for spatial feature extraction
model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Bidirectional LSTM for temporal dependencies
model.add(Bidirectional(LSTM(128, return_sequences=False)))  # Not returning sequences
model.add(Dropout(0.4))

# Fully connected layers for classification
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation='softmax'))  # 9 output classes for emotions

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

version = "2.3"
model.save(f"./project/dependencies/multimodal_sentiment/models/v{version}")