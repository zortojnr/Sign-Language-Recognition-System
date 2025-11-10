import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore


def load_data(path):
    """Load and preprocess sign language MNIST data from CSV file."""
    df = pd.read_csv(path)
    y = np.array([label if label < 9 else label - 1 for label in df['label']])
    df = df.drop('label', axis=1)
    x = np.array([df.iloc[i].to_numpy().reshape((28, 28)) for i in range(len(df))]).astype(float)
    x = np.expand_dims(x, axis=3)
    # Normalize pixel values to [0, 1]
    x = x / 255.0
    y = pd.get_dummies(y).values
    
    return x, y


# Load training data
X_train, Y_train = load_data('content/sign_mnist_train.csv')

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Display shapes
print(f"Training data shape: {X_train.shape}, Training labels shape: {Y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Validation labels shape: {Y_val.shape}")



class_names = list(string.ascii_lowercase.replace('j', '').replace('z', ''))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.xlabel(class_names[np.argmax(Y_train, axis=1)[i]])
plt.tight_layout()
plt.show()


# Model Development
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(24, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, Y_train, 
    validation_data=(X_val, Y_val), 
    epochs=10, 
    batch_size=32, 
    verbose=1
)

# Load test data (if available) or use validation set for evaluation
try:
    X_test, Y_test = load_data('content/sign_mnist_test.csv')
    print(f"Test data shape: {X_test.shape}, Test labels shape: {Y_test.shape}")
except FileNotFoundError:
    print("Test file not found. Using validation set for evaluation.")
    X_test, Y_test = X_val, Y_val

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test Accuracy: {accuracy * 100:.2f}%')