import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import tensorflow.keras as keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import time
from keras import activations
import tensorflow.keras.regularizers as regularizers
from keras.layers import BatchNormalization
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys

# Define basic values for the project
data_dir = 'data'
split_data_dir = 'split_data'
img_height = 256
img_width = 256
batch_size = 8
class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

model_folder_name = 'model_info/' + time.strftime("%Y%m%d-%H%M%S") + '/'

# Create the folder for the model
import os
os.makedirs(model_folder_name)

epochs = 20
epochs = input('Enter number of epochs: ')
# Check that the input is a number
try:
    epochs = int(epochs)
except ValueError:
    print('Please enter a number')
    sys.exit()

# Ask user if they want to add any notes
notes = input('Add any notes to the model: ')
with open(model_folder_name + 'notes.txt', 'w') as f:
    f.write(notes)

# Ask user if plots should be shown
show_plots = input('Show plots? (y/n) - default is NO: ')
if show_plots == 'y':
    show_plots = True
else:
    show_plots = False

f.close()

###
# PLOTS
###
def plot_loss_accuracy(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.savefig(model_folder_name + 'loss_accuracy.png')

    if show_plots:
        plt.show()

def plot_test_accuracy(test_images, test_preds):
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))

    for i in range(3):
        image = test_images[i].numpy().astype("uint8")
        
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Actual: {}'.format(class_names[labels[i]]))
        ax[i, 0].axis('off')
        ax[i, 1].bar(class_names, test_preds[i])
        ax[i, 1].set_title('Predicted: {}'.format(class_names[np.argmax(test_preds[i])]))

    plt.tight_layout()

    plt.savefig(model_folder_name + 'test_accuracy.png')
    if show_plots:
        plt.show()

def plot_results_bar(results, results_on_google):
    fig = plt.figure()
    plt.bar(0, results_on_google[1])
    plt.bar(1, results[1])
    plt.xticks([0, 1], ['On google images', 'On original'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')

    plt.savefig(model_folder_name + 'results_on_google_images.png')
    if show_plots:
        plt.show()

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)

    plt.savefig(model_folder_name + 'confusion_matrix_heatmap.png')
    if show_plots:
        plt.show()

def plot_class_accuracies(cm, class_names):
    accuracies = cm.diagonal() / cm.sum(axis=1)
    fig = plt.figure()
    plt.bar(class_names, accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    
    plt.savefig(model_folder_name + 'class_accuracy.png')
    if show_plots:
        plt.show()

def plot_test_images(test_images, test_labels, test_preds):
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))

    for i in range(3):
        image = test_images[i].numpy().astype("uint8")
        
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('Actual: {}'.format(class_names[test_labels[i]]))
        ax[i, 0].axis('off')
        ax[i, 1].bar(class_names, test_preds[i])
        ax[i, 1].set_title('Predicted: {}'.format(class_names[np.argmax(test_preds[i])]))

    plt.tight_layout()
    plt.savefig(model_folder_name + 'example_images.png')
    if show_plots:
        plt.show()


###
# LOADING DATA
###
def load_data(directory, img_height, img_width, batch_size):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

###
# MODEL
###

def create_preprocessing_layers():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1, 
            width_factor=0.1,
            fill_mode='nearest',
            # interpolation='bilinear',
            # seed=None,
            # fill_value=0.0,
        ),
        tf.keras.layers.RandomCrop(200, 200),
        tf.keras.layers.RandomContrast(0.2)
    ])

def create_resize_and_rescale_layers():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(img_height, img_width),
        tf.keras.layers.Rescaling(1./255)
    ])

def create_resize_and_rescale_layers_small():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(64, 64),
        tf.keras.layers.Rescaling(1./255)
    ])

# Create a tf.data pipeline of augmented images (and their labels)
def prepare_dataset(dataset, augment=False):
    preprocessing_layers = create_preprocessing_layers()
    resize_and_rescale = create_resize_and_rescale_layers()
    
    # If augment is True, apply the preprocessing_layers to the images
    if augment:
        dataset = dataset.map(lambda x, y: (preprocessing_layers(x, training=True), y))
        return dataset.map(lambda x, y: (resize_and_rescale(x), y))
    else:
        return dataset.map(lambda x, y: (resize_and_rescale(x), y))

def apply_preprocessing_layers(dataset):
    preprocessing_layers = create_preprocessing_layers()
    
    return dataset.map(lambda x,y: (preprocessing_layers(x, training=True), y))

def show_augmented_images(dataset):
    preprocessing_layers = create_preprocessing_layers()
    resize_and_rescale = create_resize_and_rescale_layers()

    for images, labels in dataset.take(1):
        for i in range(9):
            augmented_images = preprocessing_layers(images)
            # resize and rescale
            augmented_images = resize_and_rescale(augmented_images)

            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy())
            plt.axis('off')

    plt.savefig(model_folder_name + 'augmented_images.png')
    if show_plots:
        plt.show()


########################################
## LOAD DATA
# data_folder = 'split_data_20240515-182012'
data_folder = 'new_split'

google_images_data_folder = 'split_data_extra'

train_dataset = load_data(data_folder + '/training', img_height, img_width, batch_size)
validation_dataset = load_data(data_folder + '/validation', img_height, img_width, batch_size)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_folder + '/test',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Extra data from google images
google_train_dataset = load_data(google_images_data_folder + '/training', img_height, img_width, batch_size)
google_validation_dataset = load_data(google_images_data_folder + '/validation', img_height, img_width, batch_size)

google_test_dataset = tf.keras.utils.image_dataset_from_directory(
    google_images_data_folder + '/test',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Save variables to a file
with open(model_folder_name + 'data_folders.txt', 'w') as f:
    f.write(f'data_folder: {data_folder}\n')
    f.write(f'google_images_data_folder: {google_images_data_folder}\n')
    f.write(f'img_height: {img_height}\n')
    f.write(f'img_width: {img_width}\n')
    f.write(f'batch_size: {batch_size}\n')
    f.write(f'class_names: {class_names}\n')

f.close()

show_augmented_images(train_dataset)

# Preprocess the dataset first with the preprocessing layers so we can see the augmented images
# train_dataset = apply_preprocessing_layers(train_dataset)
# folder_name = f'{model_folder_name}training_augmented/'
# os.makedirs(folder_name, exist_ok=True)
    
# # Save all images in trainingset to folder 'training_augmented'
# resize_and_rescale = create_resize_and_rescale_layers()
# train_dataset = train_dataset.map(lambda x, y: (resize_and_rescale(x), y))

# for images, labels in train_dataset:
#     for i in range(images.shape[0]):
#         image = images[i].numpy()
#         label = labels[i].numpy()

#         # Create folder for each class
#         os.makedirs(f'{folder_name}{class_names[label]}', exist_ok=True)

#         plt.imsave(f'{folder_name}{class_names[label]}/{i}.png', image)
        
# Resize all images fist
def resize_images(dataset):
    resizer = Sequential([
        tf.keras.layers.Resizing(img_height, img_width)
    ])

    return dataset.map(lambda x, y: (resizer(x), y))

train_dataset = resize_images(train_dataset)
validation_dataset = resize_images(validation_dataset)
test_dataset = resize_images(test_dataset)

# BUILDING THE MODEL
model = Sequential([
    # create_preprocessing_layers(),
    # create_resize_and_rescale_layers(),
    Input(shape=(img_height, img_width, 3)),
    create_preprocessing_layers(),
    tf.keras.layers.Rescaling(1./255),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)
    ),
    Dense(128, 
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)
    ),
    Dropout(0.4),
    Dense(64,
        activation='relu',
        kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)
    ),
    Dropout(0.4),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-4,
        # beta_1=0.9,
        # beta_2=0.999,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="sc_accuracy", dtype=None),
        'accuracy',
        # precision,
        # recall
    ]
)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    # callbacks=[tensorboard_callback]
)

# Model summary
summary = model.summary()

# Save the model summary to a file
with open(model_folder_name + 'model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

f.close()


plot_loss_accuracy(history)

# EVALUATE MODEL
original_results = model.evaluate(test_dataset)
print(f'ACCURACY (rounded): {round(original_results[1], 2)}')
print(f'LOSS (rounded): {round(original_results[0], 2)}')

original_results_on_google = model.evaluate(google_test_dataset)
print(f'ACCURACY (rounded) on google images: {round(original_results_on_google[1], 2)}')
print(f'LOSS (rounded) on google images: {round(original_results_on_google[0], 2)}')

# Save the results to a file
with open(model_folder_name + 'results.txt', 'w') as f:
    f.write(f'ACCURACY: {original_results[1]}\n')
    f.write(f'LOSS: {original_results[0]}\n')
    f.write(f'ACCURACY on google images: {original_results_on_google[1]}\n')
    f.write(f'LOSS on google images: {original_results_on_google[0]}\n')

f.close()

plot_results_bar(original_results, original_results_on_google)

## Per class metrics
# Collect all labels and predictions
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())  # Collect true labels
    y_pred.extend(np.argmax(preds, axis=1))  # Collect predicted labels

# Convert lists to numpy arrays for sklearn functions
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print('--- ORIGINAL MODEL ---')
# Confusion Matrix
cm_original = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm_original)
cr_original = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", cr_original)

# Save the confusion matrix and classification report to a file
with open(model_folder_name + 'confusion_matrix.txt', 'w') as f:
    f.write(f'Confusion Matrix:\n{cm_original}\n')
    f.write(f'Classification Report:\n{cr_original}\n')

f.close()

plot_confusion_matrix(cm_original, class_names)
plot_class_accuracies(cm_original, class_names)


# Get some test images and labels, and num images in batch
test_images, test_labels = next(iter(test_dataset))
num_images = test_images.shape[0] 

# Assert that the number of images in the batch > 3
assert num_images > 3, "Plotting 3 images but batch contains less."

# Predict the batch
test_preds = model.predict(test_images)

# Plot some images and their predictions
plot_test_images(test_images, test_labels, test_preds)

# Save the model
current_date_time = time.strftime("%Y%m%d-%H%M%S")
model.save(model_folder_name + 'model' + '.keras')