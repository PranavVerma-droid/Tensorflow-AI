# Training Fashion MNIST using CNN

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')

args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Define batch size
BATCH_SIZE = 32

# Load the dataset
datasets, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True)

# Normalize and batch process the dataset
ds_train = datasets['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)


# Build the Convolutional Neural Network
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
# Here, the CNN is configured to process inputs of shape (28, 28, 1), which is the format of Fashion MNIST images
# You start with a stack of Conv2D and MaxPooling2D layers.

# Conv2D layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
# The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). 
# The next argument specifies the size of the Convolution, in this case, a 3x3 grid.
# `relu` is used as the activation function, which you might recall is the equivalent of returning x when x>0, else returning 0.

# MaxPooling2D layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
# By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.

# Dense layers
# The output from the convolution stack is flattened and passed to a set of dense layers for classification.
model = tf.keras.models.Sequential([                               
      tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])



# Train and save the model

MODEL_DIR = os.getenv("AIP_MODEL_DIR")

model.fit(ds_train, epochs=args.epochs)
model.save(MODEL_DIR)
