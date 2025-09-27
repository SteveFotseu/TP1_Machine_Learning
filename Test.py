# test.py
from tensorflow import keras
import numpy as np

# Charger MNIST (seulement x_test et y_test)
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(10000, 784)

# Charger le modèle sauvegardé
model = keras.models.load_model("mnist_model.h5")
print("Modèle chargé avec succès !")

# Tester sur 10 images
num_images = 10
predictions = model.predict(x_test[:num_images])
predicted_labels = np.argmax(predictions, axis=1)

print(f"Labels prédits pour {num_images} images :", predicted_labels)
print(f"Labels réels pour {num_images} images   :", y_test[:num_images])
