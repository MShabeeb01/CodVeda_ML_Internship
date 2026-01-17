import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Turns off the info messages

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt

# 1. LOAD DATASET FROM CSV
# Replace 'mnist_train.csv' with your actual filename
df = pd.read_csv(r"C:\Users\Samsung\Downloads\mnist_test.csv.zip")

# 2. PREPROCESS DATA
# Extract labels (y) and pixel data (X)
y = df.iloc[:, 0].values # First column is the digit (0-9)
X = df.iloc[:, 1:].values # Rest are the pixels

# Normalize pixels (0-255 -> 0-1)
X = X / 255.0

# Reshape 1D pixel rows (784) back into 2D images (28x28) for the network
X = X.reshape(-1, 28, 28)

# Split into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DESIGN ARCHITECTURE (Objective: Input, Hidden, Output)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Input Layer
    layers.Dense(128, activation='relu'), # Hidden Layer
    layers.Dense(10, activation='softmax') # Output Layer (10 digits)
])

# 

# 4. TRAIN MODEL (Objective: Backpropagation)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 5. VISUALIZE RESULTS
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Performance')
plt.legend()
plt.show()
