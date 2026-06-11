import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
# Load MNIST Dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Preprocessing - Normalize to [0,1] and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(f"Training shape: {x_train.shape}, Test shape: {x_test.shape}")
# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
model.summary()
# Compile Model
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])
# Train Model
print("\nTraining CNN model...")
history = model.fit(x_train, y_train, 
                    epochs=5, 
                    batch_size=128, 
                    validation_data=(x_test, y_test),
                    verbose=1)
# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
# Save Model
model.save("mnist_cnn_model.h5")
print("Model saved as mnist_cnn_model.h5")
