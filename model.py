import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Define basic values for the project
data_dir = 'data'
img_height = 256
img_width = 256
batch_size = 16
class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

data = tf.keras.utils.image_dataset_from_directory(
    'data', 
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    # class_names=class_names
)

# Print class names
print(data.class_names)

# Pre-processing
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
images, labels = next(scaled_iterator)

training_size = int(0.7*len(data))
validation_size = int(0.2*len(data))+1
test_size = int(0.1*len(data))

print(training_size+validation_size+test_size == len(data))

training_set = data.take(training_size)
validation_set = data.skip(training_size).take(validation_size)
test_set = data.skip(training_size+validation_size).take(test_size)


# Build model
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()

# First input layer, convolution has 16 filters, each filter is 3x3
# Activation function is ReLU... why??
# Stride is 1
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
# Pooling layer, 2x2 default stride, takes max values from 2x2 grid and passes them on so halves the size
model.add(MaxPooling2D((2, 2)))

# Second convolutional layer, 32 filters, 3x3
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
# Second pooling layer
model.add(MaxPooling2D((2, 2)))

# Third convolutional layer, 16 filters, 3x3
model.add(Conv2D(16, (3, 3), 1, activation='relu'))

# model.add(Conv2D(64, (3, 3), 1, activation='relu')) --> everything is a hammer

# Third pooling layer
model.add(MaxPooling2D((2, 2)))

# Flatten the data, we dont want channel values, so we flatten
model.add(Flatten())

# First fully connected layer, 256 neurons
model.add(Dense(256, activation='relu'))

# Output layer, 4 neurons, one for each class
model.add(Dense(4, activation='softmax'))

# # Dropout layer, 0.2 probability of dropout, not needed.. check loss values for overfitting
# model.add(Dropout(0.2))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    loss=keras.losses.SparseCategoricalCrossentropy(), 
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="sc_accuracy", 
            dtype=None
        ),
        # https://keras.io/api/metrics/classification_metrics/#precision-class
        # If used with a loss function that sets from_logits=True 
        # (i.e. no sigmoid applied to predictions), thresholds should be set to 0. 
        # from_logits=False because we apply a softmax activation function to the output layer
        keras.metrics.Precision(
            thresholds=None, 
            top_k=None, 
            class_id=None, 
            name='precision', 
            dtype=None
        ),
        keras.metrics.Recall(
            thresholds=None, 
            top_k=None, 
            class_id=None, 
            name='recall', 
            dtype=None
        ),
        # keras.metrics.F1Score(
        #     average="micro", 
        #     threshold=None, 
        #     name="f1_score", 
        #     dtype=None
        # ),
    ]
)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(
    training_set, 
    validation_data=validation_set, 
    epochs=20, 
    callbacks=[tensorboard_callback]
)