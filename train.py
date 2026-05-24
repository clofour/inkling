from shared import DATA_DIR, MODEL_DIR, CATEGORIES, IMAGE_SIZE
import random
from datetime import datetime
import os.path as path
import numpy as np
import tensorflow as tf
from keras import models, layers, losses
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DRAWING_COUNT = 10000
VALIDATION_FRACTION = 0.2

date = datetime.now().strftime(r"%Y%m%d_%H%M")

def load_data():
    data = {}

    for category in CATEGORIES:
        category_file_path = path.join(DATA_DIR, f"{category}.npy")
        category_data = np.load(file=category_file_path)
        data[category] = category_data

    return data

def visualize_data(data):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        chosen_category = random.choice(CATEGORIES)
        bitmap = data[chosen_category][i]
        image = bitmap.reshape(IMAGE_SIZE, IMAGE_SIZE)

        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(chosen_category)
    
    plt.show()

def process_data(data):
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

    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    images = images / 255.0

    permutation = np.random.permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]

    data_split_index = int(len(images) * VALIDATION_FRACTION)
    x_training = images[:data_split_index]
    y_training = labels[:data_split_index]
    x_validation = images[data_split_index:]
    y_validation = labels[data_split_index:]

    return x_training, y_training, x_validation, y_validation

def create_model():
    model = models.Sequential()
    model.add(layers.Input((IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(CATEGORIES)))
    model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model

def train_model(model, x_training, y_training, x_validation, y_validation):
    training_info = model.fit(x=x_training, y=y_training, epochs=10, validation_data=(x_validation, y_validation))
    model.save(path.join(MODEL_DIR, f"{date}.keras"))

    return training_info

def visualize_results(model, x_training, y_training, x_validation, y_validation, training_info):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_axis = axes[0]
    accuracy_axis = axes[1]

    y_prediction = model.predict(x_validation)
    y_prediction = np.argmax(y_prediction, axis=1)
    matrix = confusion_matrix(y_validation, y_prediction)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CATEGORIES)
    matrix_display.plot(ax=cm_axis, colorbar=False)
    cm_axis.set_title("Confusion Matrix")

    accuracy_axis.plot(training_info.history["accuracy"], label="accuracy")
    accuracy_axis.plot(training_info.history["val_accuracy"], label="val_accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim([0.5, 1])
    accuracy_axis.legend(loc="lower right")
    accuracy_axis.set_title("Accuracy")

    plt.show()

data = load_data()
visualize_data(data)
x_training, y_training, x_validation, y_validation = process_data(data)
model = create_model()
training_info = train_model(model, x_training, y_training, x_validation, y_validation)
visualize_results(model, x_training, y_training, x_validation, y_validation, training_info)
