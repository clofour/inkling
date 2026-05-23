import random
import os.path as path
import numpy as np
import tensorflow as tf
from keras import models, layers, losses
import matplotlib.pyplot as plt

DATA_DIR = "./data"
CATEGORIES = [
    "apple",
    "carrot",
    "cat",
    "house",
    "umbrella"
]
DRAWING_COUNT = 1000
VALIDATION_FRACTION = 0.2

data = {}

for category in CATEGORIES:
    category_file_path = path.join(DATA_DIR, f"{category}.npy")
    category_data = np.load(file=category_file_path)
    data[category] = category_data

plt.figure(figsize=(10, 10))
for i in range(25):
    chosen_category = random.choice(CATEGORIES)
    bitmap = data[chosen_category][i]
    image = bitmap.reshape(28, 28)

    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.xlabel(chosen_category)
plt.show()

images = []
labels = []

for label, category in enumerate(CATEGORIES):
    category_data = data[category]
    images.append(category_data[:DRAWING_COUNT])
    labels.append(np.full(DRAWING_COUNT, label))

images = np.concatenate(images)
labels = np.concatenate(labels)

images = np.array(images)
labels = np.array(labels)

images = images.reshape(-1, 28, 28, 1)
images = images / 255.0

permutation = np.random.permutation(len(images))
images = images[permutation]
labels = labels[permutation]

data_split_index = int(len(images) * VALIDATION_FRACTION)
x_training = images[:data_split_index]
y_training = labels[:data_split_index]
x_validation = images[data_split_index:]
y_validation = labels[data_split_index:]

model = models.Sequential()
model.add(layers.Input((28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(len(CATEGORIES)))
model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
history = model.fit(x=x_training, y=y_training, epochs=10, validation_data=(x_validation, y_validation))

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

# test_loss, test_acc = model.evaluate()
