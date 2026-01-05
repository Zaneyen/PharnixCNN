import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from random import randint

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_train_data(x_train, y_train, x_test, y_test):
    # Reshape : ajouter dimension canal
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')
    # One-hot pour label smoothing
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_mnist()
X_train.shape, y_train.shape, X_test.shape, y_test.shape

@keras.saving.register_keras_serializable()
class SimpleCNN_MNIST(Model):
    """
    CNN optimisé pour MNIST

    Architecture:
        - Data Augmentation (rotation, translation, zoom)
        - Conv 32 (3×3) → BN → ReLU → MaxPool (28→14)
        - Conv 64 (3×3) → BN → ReLU → MaxPool (14→7)
        - Conv 128 (3×3) → BN → ReLU
        - Conv 256 (3×3) → BN → ReLU
        - GlobalAveragePooling
        - Dropout → Dense 10

    ~400K paramètres, cible 99.4%+
    """

    def __init__(self, num_classes=10, dropout_rate=0.3, mu=33.3184, std=78.5675):
        super().__init__()

        # Sauvegarder pour get_config
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mu = mu
        self.std_val = std

        # Normalisation (tenseurs pour le calcul)
        self.mean = tf.constant(mu, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

        # Data Augmentation
        self.augmentation = keras.Sequential([
            layers.RandomRotation(0.05),        # ±18°
            layers.RandomTranslation(0.1, 0.1), # ±10% shift
            layers.RandomZoom(0.1),             # ±10% zoom
        ])

        # Bloc 1 : 28×28×1 → 14×14×32
        self.conv1 = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)

        # Bloc 2 : 14×14×32 → 7×7×64
        self.conv2 = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)

        # Bloc 3 : 7×7×64 → 7×7×128
        self.conv3 = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()

        # Bloc 4 : 7×7×128 → 7×7×256
        self.conv4 = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')
        self.bn4 = layers.BatchNormalization()

        # Classification
        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        # Data augmentation seulement à l'entraînement
        if training:
            x = self.augmentation(x)

        # Normalisation
        x = (x - self.mean) / self.std

        # Bloc 1
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.pool1(x)

        # Bloc 2
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = self.pool2(x)

        # Bloc 3
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))

        # Bloc 4
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))

        # Classification
        x = self.gap(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        return x

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'mu': self.mu,
            'std': self.std_val
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
(X_train_processed, y_train_processed), (X_test_processed, y_test_processed) = preprocess_train_data(
    X_train, y_train,
    X_test, y_test
  )

model = SimpleCNN_MNIST(mu = X_train.mean(), std = X_train.std())
dummy_input = tf.zeros((1, 28, 28, 1))
_ = model(dummy_input)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

# Entraînement avec suivi sur test
history = model.fit(
    X_train_processed, y_train_processed,
    batch_size=128,
    epochs=20,
    validation_data=(X_test_processed, y_test_processed),
    callbacks=callbacks
)

model.evaluate(X_test_processed, y_test_processed)

# Sauvegarder le modèle complet
model.save('mnist_cnn.keras')