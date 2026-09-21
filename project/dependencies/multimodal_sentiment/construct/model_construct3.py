from keras import Model, Input
from keras.layers import (
    Dense, Conv1D, LSTM, Bidirectional, Dropout, Flatten,
    GlobalAveragePooling1D, Attention, Concatenate, LayerNormalization,
    MultiHeadAttention, Layer, BatchNormalization
)
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K


# Custom Focal Loss for Imbalanced Data
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.clip(y_true, epsilon, 1. - epsilon)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed


# Build MFCC processing path
mfcc_input = Input(shape=(40, 1), name="mfcc_data")
conv1 = Conv1D(64, 3, padding="same", activation="relu")(mfcc_input)
conv1 = Dropout(0.25)(BatchNormalization()(conv1))
conv2 = Conv1D(128, 3, padding="same", activation="relu")(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Dropout(0.25)(BatchNormalization()(conv2))
flatten_mfcc = Flatten()(conv2)




# Apply Attention Layer
#mfcc_attn = Attention()([flatten_mfcc, flatten_text])
#mfcc_attn = Dropout(0.3)(mfcc_attn)

# Continue with the rest of the architecture
#combined = Concatenate()([mfcc_attn, flatten_mfcc, flatten_text])

# Dense Layers for Decision Making
# Add L2 regularization to the dense layers
dense1 = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(flatten_mfcc)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(dropout1)
output = Dense(9, activation="softmax", name="output")(dense2)


# Model Assembly
model = Model(inputs=[mfcc_input], outputs=output)

# Compile the model with a focal loss
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

# Model summary
model.summary()

# Save the model
version = "2.0.2"
model.save(f"./project/dependencies/multimodal_sentiment/models/v{version}")