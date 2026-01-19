#project by Pedro Leite (201906697) and Rubén Pombo (202302830)
#python3 project.py
#uncomment the function with the model that you want to test (at the bottom of the file)
import time
import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, History
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.regularizers import l2

#download data
#https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

#read the data
df = pd.read_csv('data/HAM10000_metadata.csv')

#remove NULL observations
df.dropna(axis=0, inplace=True)
#remove observations = 0
df = df[df['age']!=0]
#remove unknown observations
for column in df.columns:
    df = df[df[column]!='unknown']

#create a dictionary: {file_name: path_to_file}
image_dict = {}
for i in os.listdir("Data/HAM10000_images_part_1"):
    file_name = os.path.splitext(i)[0]
    image_dict[file_name] = "Data/HAM10000_images_part_1/"+i
for i in os.listdir("Data/HAM10000_images_part_2"):
    file_name = os.path.splitext(i)[0]
    image_dict[file_name] = "Data/HAM10000_images_part_2/"+i

#add path_to_file to the dataframe
df['path'] = df['image_id'].map(image_dict.get)
#add cell_type to the dataframe, this are going to be our target variables
cell_type_dict = {
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}
df['cell_type'] = df['dx'].map(cell_type_dict.get)
#add cell_type_index (nv = 1, mel = 2, ...) to the dataframe
df['cell_type_index'] = pd.Categorical(df['cell_type']).codes
#add numpy array to resize the image to 125x100 (so it is easier to apply the CNNs)
#to the dataframe
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((125,100))))

#our target variable is cell_type_index
features = df.drop(columns=['cell_type_index'], axis=1)
target = df['cell_type_index']

#divide our data into training = 75% and testing = 25%
features_train_initial, features_test_initial, targets_train_initial, targets_test_initial = train_test_split(features, target, test_size=0.25, random_state=123)

#convert the images to an np array
features_train = np.asarray(features_train_initial['image'].tolist())
features_test = np.asarray(features_test_initial['image'].tolist())

#normalize the images, in order to have a faster model
features_train = (features_train-np.mean(features_train))/np.std(features_train)
features_test = (features_test-np.mean(features_test))/np.std(features_test)

#one-hot encoding the target variables, because we want multi-classification
targets_train = to_categorical(targets_train_initial, num_classes=7)
targets_test = to_categorical(targets_test_initial, num_classes=7)

#divide our data into training = 90% and validation = 10% (from the training dataset)
features_train, features_validate, targets_train, targets_validate = train_test_split(features_train, targets_train, test_size=0.1, random_state=123)

#reshape the images from 3 dimensions to 2 (100x25 is the size of the image, 3 is the colors
#channels RGB)
features_train = features_train.reshape(features_train.shape[0], *(100,125,3))
features_test = features_test.reshape(features_test.shape[0], *(100,125,3))
features_validate = features_validate.reshape(features_validate.shape[0], *(100,125,3))

def test(start_time, model, training, features_test, targets_test, features_validate, targets_validate):
    #test the model
    loss_test, accuracy_test = model.evaluate(features_test, targets_test, verbose=1)
    loss_validation, accuracy_validation = model.evaluate(features_validate, targets_validate, verbose=1)
    end_time = time.time()

    #results
    print("Time (50 Epochs):", int((end_time-start_time)/60), "minutes")
    print("Accuracy (Test):", int(accuracy_test*100), "%")
    print("Loss (Test):", int(loss_test*100), "%")
    print("Accuracy (Validation):", int(accuracy_validation*100), "%")
    print("Loss (Validation):", int(loss_validation*100), "%")

    #size
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    #accuracy
    axs[0].plot(range(1, len(training.history['accuracy']) + 1), training.history['accuracy'])
    #validation accuracy
    axs[0].plot(range(1, len(training.history['val_accuracy']) + 1), training.history['val_accuracy'])
    #labels
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #ticks
    axs[0].set_xticks(np.arange(1, len(training.history['accuracy']) + 1, len(training.history['accuracy']) / 10))
    #labels
    axs[0].legend(['Train', 'Validation'], loc='best')

    #loss
    axs[1].plot(range(1, len(training.history['loss']) + 1), training.history['loss'])
    #validation loss
    axs[1].plot(range(1, len(training.history['val_loss']) + 1), training.history['val_loss'])
    #labels
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    #ticks
    axs[1].set_xticks(np.arange(1, len(training.history['loss']) + 1, len(training.history['loss']) / 10))
    #labels
    axs[1].legend(['Train', 'Validation'], loc='best')

    #plot
    plt.show()

def nn1(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #reshape the images from 3 dimensions to 2 (100x25 is the size of the image, 3 is the colors
    #channels RGB)
    features_train = features_train.reshape(features_train.shape[0], 100*125*3)
    features_test = features_test.reshape(features_test.shape[0], 100*125*3)
    features_validate = features_validate.reshape(features_validate.shape[0], 100*125*3)

    #our nn model with 4 hidden layers
    model = Sequential()
    #the input layer and the hidden layers have 64 neurons and the relu activation function
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=100*125*3))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function because thats the appropriate for multiclass classification
    model.add(Dense(units=7, kernel_initializer='uniform', activation='softmax'))

    #adapte the learning rate for each parameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00075, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    #train the model
    training = model.fit(features_train, targets_train, validation_data=(features_validate, targets_validate),
                        batch_size=10, epochs=50)

    #Time (50 Epochs): 15 minutes
    #Accuracy (Test): 69% Loss (Test): 202%
    #Accuracy (Validation): 72% Loss (Validation): 163%
    test(start_time, model, training, features_test, targets_test, features_validate, targets_validate)

def cnn1(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #our cnn: 3 hidden layers, 1 maxpooling layer, 1 dropout layer
    #(the same repeats 2 more times), 1 flatten layer, 2 fully connected layers,
    #1 dropout layer, 1 output layer
    dropout = 0.15
    model = Sequential()
    for i in range(3):
        #the input layer and the hidden layers have 32 neurons, the relu activation function
        #and a filter of 3x3, we also add padding to assure that the input image is the same
        #size as the output image
        if i == 0:
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same', input_shape=(100, 125, 3)))
        else:
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same'))
        #maxpooling to reduce data dimensionality
        model.add(MaxPool2D(pool_size=(2,2)))
        #dropout to avoid overfitting
        model.add(Dropout(dropout))
        dropout = dropout+(dropout*0.5)
    #flatten layer to convert the 3d input to an array, that will be usefull
    #for the classification
    model.add(Flatten())
    #fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #dropout to avoid overfitting
    model.add(Dropout(dropout))
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    model.add(Dense(units=7, activation='softmax'))

    #adapte the learning rate for each parameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    start_time = time.time()
    #train the model
    training = model.fit(features_train, targets_train, validation_data=(features_validate, targets_validate),
                        batch_size=10, epochs=50)

    #Time (50 Epochs): 58 minutes
    #Accuracy (Test): 75% Loss (Test): 106%
    #Accuracy (Validation): 76% Loss (Validation): 98%
    test(start_time, model, training, features_test, targets_test, features_validate, targets_validate)

def cnn2(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #our cnn: 3 hidden layers, 1 maxpooling layer, 1 dropout layer
    #(the same repeats 2 more times), 1 flatten layer, 2 fully connected layers,
    #1 dropout layer, 1 output layer
    dropout = 0.15
    model = Sequential()
    for i in range(3):
        #the input layer and the hidden layers have 32 neurons, the relu activation function
        #and a filter of 3x3, we also add padding to assure that the input image is the same
        #size as the output image
        if i == 0:
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same', input_shape=(100, 125, 3)))
        else:
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same'))
        #maxpooling to reduce data dimensionality
        model.add(MaxPool2D(pool_size=(2,2)))
        #dropout to avoid overfitting
        model.add(Dropout(dropout))
        dropout = dropout+(dropout*0.5)
    #flatten layer to convert the 3d input to an array, that will be usefull
    #for the classification
    model.add(Flatten())
    #fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #dropout to avoid overfitting
    model.add(Dropout(dropout))
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    model.add(Dense(units=7, activation='softmax'))

    #adapte the learning rate for each parameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #reduce learning rate if its not improving (through the epochs = 5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    #data augmentation to avoid overfitting, by executing several operations in the
    #training dataset, like zoom, rotation, shifts, flips, ...
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.12,
        height_shift_range = 0.12,
        horizontal_flip = True,
        vertical_flip = True)
    datagen.fit(features_train)

    start_time = time.time()
    #train the model (with data augmentation)
    training = model.fit_generator(datagen.flow(features_train, targets_train, batch_size=10),
                                  epochs=50, validation_data=(features_validate, targets_validate),
                                  verbose=1, steps_per_epoch=features_train.shape[0]//10
                                  , callbacks=[learning_rate_reduction])

    #Time (50 Epochs): 58 minutes
    #Accuracy (Test): 76% Loss (Test): 65%
    #Accuracy (Validation): 77% Loss (Validation): 61%
    test(start_time, model, training, features_test, targets_test, features_validate, targets_validate)

def mobile_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #transfer learning model
    #initial mobilenet model of the tensorflow library, but with a new input shape
    mobilenet_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(100, 125, 3))
    new_input = tf.keras.layers.Input(shape=(100, 125, 3))
    feature_layers = mobilenet_model(new_input)

    #convert the model to JSON
    new_model = tf.keras.models.model_from_json(mobilenet_model.to_json(), custom_objects=None)
    #copy the weights from the old model
    for layer in new_model.layers:
        layer.set_weights(mobilenet_model.get_layer(name=layer.name).get_weights())

    #add a maxpooling layer to our model
    feature_layers = tf.keras.layers.GlobalAveragePooling2D()(feature_layers)
    #dropout to avoid overfitting
    feature_layers = tf.keras.layers.Dropout(0.3)(feature_layers)
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    output_layer = tf.keras.layers.Dense(units=7, activation='softmax')(feature_layers)

    #update our model
    new_model = tf.keras.models.Model(inputs=new_input, outputs=output_layer)

    #we will train the last 23 layers
    for layer in mobilenet_model.layers[:-23]:
        layer.trainable = False

    #adapte the learning rate for each parameter
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #reduce learning rate if its not improving (through the epochs = 5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    #data augmentation to avoid overfitting, by executing several operations in the
    #training dataset, like zoom, rotation, shifts, flips, ...
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.12,
        height_shift_range = 0.12,
        horizontal_flip = True,
        vertical_flip = True)
    datagen.fit(features_train)

    start_time = time.time()
    #train the model (with data augmentation)
    training = new_model.fit_generator(datagen.flow(features_train, targets_train, batch_size=10),
                                       epochs=50, validation_data=(features_validate, targets_validate),
                                       verbose=1, steps_per_epoch=features_train.shape[0]//10
                                       , callbacks=[learning_rate_reduction])

    #Time (50 Epochs):  54 minutes
    #Accuracy (Test): 81 %  Loss (Test): 60 %
    #Accuracy (Validation):  83% Loss (Validation): 58%
    test(start_time, new_model, training, features_test, targets_test, features_validate, targets_validate)

def res_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #transfer learning model
    #initial resnet model of the tensorflow library, but with a new input shape
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 125, 3))
    new_input = tf.keras.layers.Input(shape=(100, 125, 3))
    feature_layers = resnet_model(new_input)

    #convert the model to JSON
    new_model = tf.keras.models.model_from_json(resnet_model.to_json(), custom_objects=None)
    #copy the weights from the old model
    for layer in new_model.layers:
        layer.set_weights(resnet_model.get_layer(name=layer.name).get_weights())

    #add a maxpooling layer to our model
    feature_layers = tf.keras.layers.GlobalAveragePooling2D()(feature_layers)
    #dropout to avoid overfitting
    feature_layers = tf.keras.layers.Dropout(0.3)(feature_layers)
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    output_layer = tf.keras.layers.Dense(units=7, activation='softmax')(feature_layers)

    #update our model
    new_model = tf.keras.models.Model(inputs=new_input, outputs=output_layer)

    #we will train the last 23 layers
    for layer in resnet_model.layers[:-23]:
        layer.trainable = False

    #adapte the learning rate for each parameter
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #reduce learning rate if its not improving (through the epochs = 5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    #data augmentation to avoid overfitting, by executing several operations in the
    #training dataset, like zoom, rotation, shifts, flips, ...
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.12,
        height_shift_range = 0.12,
        horizontal_flip = True,
        vertical_flip = True)
    datagen.fit(features_train)

    start_time = time.time()
    #train the model (with data augmentation)
    training = new_model.fit_generator(datagen.flow(features_train, targets_train, batch_size=10),
                                       epochs=50, validation_data=(features_validate, targets_validate),
                                       verbose=1, steps_per_epoch=features_train.shape[0]//10
                                       , callbacks=[learning_rate_reduction])

    #Time (50 Epochs): 217 minutes
    #Accuracy (Test): 75% Loss (Test): 78%
    #Accuracy (Validation):  77%  Loss (Validation): 69%
    test(start_time, new_model, training, features_test, targets_test, features_validate, targets_validate)

def dense_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #transfer learning model
    #initial densenet model of the tensorflow library, but with a new input shape
    densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(100, 125, 3))
    new_input = tf.keras.layers.Input(shape=(100, 125, 3))
    feature_layers = densenet_model(new_input)

    #convert the model to JSON
    new_model = tf.keras.models.model_from_json(densenet_model.to_json(), custom_objects=None)
    #update the weights
    for layer in new_model.layers:
        layer.set_weights(densenet_model.get_layer(name=layer.name).get_weights())

    #add a maxpooling layer to our model
    feature_layers = tf.keras.layers.GlobalAveragePooling2D()(feature_layers)
    #dropout to avoid overfitting
    feature_layers = tf.keras.layers.Dropout(0.3)(feature_layers)
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    output_layer = tf.keras.layers.Dense(units=7, activation='softmax')(feature_layers)

    #update our model
    new_model = tf.keras.models.Model(inputs=new_input, outputs=output_layer)

    #we will train the last 23 layers
    for layer in densenet_model.layers[:-23]:
        layer.trainable = False

    #adapte the learning rate for each parameter
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #reduce learning rate if its not improving (through the epochs = 5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    #data augmentation to avoid overfitting, by executing several operations in the
    #training dataset, like zoom, rotation, shifts, flips, ...
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.12,
        height_shift_range = 0.12,
        horizontal_flip = True,
        vertical_flip = True)
    datagen.fit(features_train)

    start_time = time.time()
    #train the model (with data augmentation)
    training = new_model.fit_generator(datagen.flow(features_train, targets_train, batch_size=10),
                                       epochs=50, validation_data=(features_validate, targets_validate),
                                       verbose=1, steps_per_epoch=features_train.shape[0]//10
                                       , callbacks=[learning_rate_reduction])

    #Time (50 Epochs): 135 minutes
    #Accuracy (Test): 82%  Loss (Test): 57%
    #Accuracy (Validation):  81% Loss (Validation): 56%
    test(start_time, new_model, training, features_test, targets_test, features_validate, targets_validate)

def mobile_net2(features_train, targets_train, features_test, targets_test, features_validate, targets_validate):
    #transfer learning model
    #initial mobilenet model of the tensorflow library, but with a new input shape
    mobilenet_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(100, 125, 3))
    new_input = tf.keras.layers.Input(shape=(100, 125, 3))
    feature_layers = mobilenet_model(new_input)

    #convert the model to JSON
    new_model = tf.keras.models.model_from_json(mobilenet_model.to_json(), custom_objects=None)
    #copy the weights from the old model
    for layer in new_model.layers:
        layer.set_weights(mobilenet_model.get_layer(name=layer.name).get_weights())

    #add a maxpooling layer to our model
    feature_layers = tf.keras.layers.GlobalAveragePooling2D()(feature_layers)
    #dropout to avoid overfitting
    #BIGGER DROPOUT
    feature_layers = tf.keras.layers.Dropout(0.5)(feature_layers)
    #the output layer has 7 neurons because thats the number of targets and we use
    #the softmax function beacuse thats the appropriate for multiclass classification
    #WITH REGULARIZATION
    output_layer = tf.keras.layers.Dense(units=7, activation='softmax', kernel_regularizer=l2(0.001))(feature_layers)

    #ADD BATCH NORMALIZATION
    feature_layers = tf.keras.layers.BatchNormalization()(feature_layers)

    #update our model
    new_model = tf.keras.models.Model(inputs=new_input, outputs=output_layer)

    #we will train the last 23 layers
    #MORE LAYERS
    for layer in mobilenet_model.layers[:-50]:
        layer.trainable = False

    #adapte the learning rate for each parameter
    #LOWER THAN THE PREVIOUS ONE
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #compile the model
    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #reduce learning rate if its not improving (through the epochs = 5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

    #data augmentation to avoid overfitting, by executing several operations in the
    #training dataset, like zoom, rotation, shifts, flips, ...
    #BIGGER THAN THE PREVIOUS ONE
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 30,
        zoom_range = 0.3,
        width_shift_range = 0.12,
        height_shift_range = 0.12,
        horizontal_flip = True,
        vertical_flip = True)
    datagen.fit(features_train)

    start_time = time.time()
    #train the model (with data augmentation)
    training = new_model.fit_generator(datagen.flow(features_train, targets_train, batch_size=10),
                                       epochs=500, validation_data=(features_validate, targets_validate),
                                       verbose=1, steps_per_epoch=features_train.shape[0]//10
                                       , callbacks=[learning_rate_reduction])

    #Time (500 Epochs): 1092 minutes
    #Accuracy (Test): 83% Loss (Test): 68%
    #Accuracy (Validation): 84% Loss (Validation): 62%
    test(start_time, new_model, training, features_test, targets_test, features_validate, targets_validate)

#nn1(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#cnn1(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#cnn2(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#mobile_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#res_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#dense_net(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
#mobile_net2(features_train, targets_train, features_test, targets_test, features_validate, targets_validate)
