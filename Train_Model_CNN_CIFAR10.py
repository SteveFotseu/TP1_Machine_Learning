# ==========================
# 1️⃣ Importation des bibliothèques nécessaires
# ==========================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ==========================
# 2️⃣ Chargement de l'ensemble de données CIFAR-10
# ==========================

# CIFAR-10 contient 60 000 images 32x32 en couleur réparties sur 10 classes
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# ==========================
# 3️⃣ Définition des constantes du jeu de données
# ==========================

NUM_CLASSES = 10                                 # CIFAR-10 contient 10 classes
INPUT_SHAPE = x_train.shape[1:]                  # (32, 32, 3) → taille d'une image

# ==========================
# 4️⃣ Normalisation des valeurs de pixels
# ==========================

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ==========================
# 5️⃣ Conversion des étiquettes (labels) en One-Hot Encoding
# ==========================

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# ==========================
# 6️⃣ Affichage des dimensions pour vérification
# ==========================

print(f"Forme des données d'entrée : {INPUT_SHAPE}")
print(f"x_train : {x_train.shape}, x_test : {x_test.shape}")
print(f"y_train : {y_train.shape}, y_test : {y_test.shape}\n")

# =======================================================
# 7️⃣ Définition de la fonction du modèle CNN (Tâche 1)
# =======================================================

def build_basic_cnn(input_shape, num_classes):
    """
    Définit un modèle CNN simple pour la classification CIFAR-10.
    Architecture : Conv(32) -> MaxPool -> Conv(64) -> MaxPool -> Flatten -> Dense(512) -> Dense(10)
    """
    model = keras.Sequential([
        # Couche Convolutive 1: 32 filtres 3x3, activation ReLU
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),

        # Couche de Pooling 1: Pooling Maximal 2x2
        layers.MaxPooling2D(pool_size=(2, 2)), # <-- Complété le TODO 1

        # Couche Convolutive 2: 64 filtres 3x3, activation ReLU
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # <-- Complété le TODO 2

        # Couche de Pooling 2: Pooling Maximal 2x2
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Aplatir le calque pour passer aux calques denses
        layers.Flatten(),

        # Couche Dense 1: 512 unités, activation ReLU
        layers.Dense(512, activation='relu'),

        # Couche de Sortie: NUM_CLASSES unités, activation Softmax
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# =======================================================
# 8️⃣ Compilation et Entraînement du modèle (Tâche 2)
# =======================================================

model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Début de l'entraînement du modèle...")
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10, # Entraîner pendant au moins 10 époques comme demandé
    validation_split=0.1  # 10 % des données d'entraînement pour la validation
)
print("Fin de l'entraînement.")

# =======================================================
# 9️⃣ Évaluation finale sur le jeu de test
# =======================================================

print("\nÉvaluation sur le jeu de test (x_test, y_test) :")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"\nPrécision finale sur le jeu de test : {test_acc:.4f}")