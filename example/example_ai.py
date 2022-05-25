import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger

checkpoint_path = "checkpoints/first_data.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Build the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input, Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer, 128 neurons, relu activation
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer, 10 neurons, softmax activation
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model


# Compile the model
# model.summary()


# Train the model
def train_model():
    model = create_model()
    model.fit(train_images, train_labels, epochs=50, callbacks=[cp_callback])


def test_model():
    model = create_model()
    model.load_weights(checkpoint_path)

    # Test the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    logger.success("Restored model, accuracy: {:5.2f}%".format(100 * test_acc))


def predict_image(image_number):
    model = create_model()
    model.load_weights(checkpoint_path)

    prediction = np.argmax(model.predict(test_images)[image_number])

    logger.success(f"Prediction: class_num:{prediction} class_name:{class_names[prediction]}")

    logger.success("Right? {}".format(test_labels[image_number] == prediction))

    plt.figure()
    plt.imshow(test_images[image_number])
    plt.colorbar()
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    # train_model()
    # test_model()
    predict_image(random.randint(0, 9999))
    logger.success("Done!")
