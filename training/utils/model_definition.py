"""
Définition du modèle SimpleCNN_MNIST pour l'entraînement
"""
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model


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

    ~300K paramètres, cible 99.5%+
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