## DATA PREPROCESSING ##
# Imports
import glob
import numpy as np
import os.path as path
from scipy import misc

# IMAGE_PATH should be the path to the planesnet folder
IMAGE_PATH = 'data\planesnet'
file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))

# Load the images
images = [misc.imread(path) for path in file_paths]
images = np.asarray(images)

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

# Scale
images = images / 255

# Read the labels from the filenames
n_images = images.shape[0]
labels = np.zeros(n_images)
for i in range(n_images):
    filename = path.basename(file_paths[i])[0]
    labels[i] = int(filename[0])

# Split into test and training sets
TRAIN_TEST_SPLIT = 0.9

# Split at the given index
split_index = int(TRAIN_TEST_SPLIT * n_images)
shuffled_indices = np.random.permutation(n_images)
train_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]

# Split the images and the labels
x_train = images[train_indices, :, :]
y_train = labels[train_indices]
x_test = images[test_indices, :, :]
y_test = labels[test_indices]

## DATA VISUALIZATION ##
# Define data visualization function
VISUALIZE_DATA = False
if VISUALIZE_DATA:
    import matplotlib.pyplot as plt

    def visualize_data(positive_images, negative_images):
        # INPUTS
        # positive_images - Images where the label = 1 (True)
        # negative_images - Images where the label = 0 (False)

        figure = plt.figure()
        count = 0
        for i in range(positive_images.shape[0]):
            count += 1
            figure.add_subplot(2, positive_images.shape[0], count)
            plt.imshow(positive_images[i, :, :])
            plt.axis('off')
            plt.title("1")

            figure.add_subplot(1, negative_images.shape[0], count)
            plt.imshow(negative_images[i, :, :])
            plt.axis('off')
            plt.title("0")
        plt.show()


    # Number of positive and negative examples to show
    N_TO_VISUALIZE = 10

    # Select the first N positive examples
    positive_example_indices = (y_train == 1)
    positive_examples = x_train[positive_example_indices, :, :]
    positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]

    # Select the first N negative examples
    negative_example_indices = (y_train == 0)
    negative_examples = x_train[negative_example_indices, :, :]
    negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]

    # Call the visualization function
    visualize_data(positive_examples, negative_examples)

## MODEL CREATION ##
# Imports
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# Model Hyperparamaters
N_LAYERS = 4

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define model hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer with dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

# Instantiate the model
model = cnn(size=image_size, n_layers=N_LAYERS)


## MODEL TRAINING ##
# Training Hyperparamters
EPOCHS = 150
BATCH_SIZE = 200

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = 'logdir'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)

## MODEL EVALUATION ##
# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)

# Report the accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))

VISUALIZE_RESULTS = True
if VISUALIZE_RESULTS:
    import matplotlib.pyplot as plt
    def visualize_incorrect_labels(x_data, y_real, y_predicted):
        # INPUTS
        # x_data      - images
        # y_data      - ground truth labels
        # y_predicted - predicted label
        count = 0
        figure = plt.figure()
        incorrect_label_indices = (y_real != y_predicted)
        y_real = y_real[incorrect_label_indices]
        y_predicted = y_predicted[incorrect_label_indices]
        x_data = x_data[incorrect_label_indices, :, :, :]

        maximum_square = np.ceil(np.sqrt(x_data.shape[0]))

        for i in range(x_data.shape[0]):
            count += 1
            figure.add_subplot(maximum_square, maximum_square, count)
            plt.imshow(x_data[i, :, :, :])
            plt.axis('off')
            plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)

        plt.show()

    visualize_incorrect_labels(x_test, y_test, np.asarray(test_predictions).ravel())
