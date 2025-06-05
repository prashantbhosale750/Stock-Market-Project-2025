import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the input data
X = np.load("data/processed/transformer_X.npy")
y = np.load("data/processed/transformer_y.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x_ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Input layer
input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)

# Transformer blocks
x = transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)

# Global pooling and output
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(1)(x)

# Compile model
model = Model(inputs, outputs)
model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/transformer_regressor.h5")
print("\n✅ Transformer model saved to models/transformer_regressor.h5")
