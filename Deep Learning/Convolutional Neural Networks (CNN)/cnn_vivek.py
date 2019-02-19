# CNN

# Building CNN

# Importing Libraries for keras

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense

# Initializing CNN
classifier = Sequential()

# 1 - Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

# 2 - Pooling Layer
classifier.add(MaxPool2D(pool_size=(2,2)))

# Adding Second Convolution Layer and Pooling Layer
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# Adding Third Convolution Layer and Pooling Layer
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))



# 3 - Flattening
classifier.add(Flatten())

# 4 - Fully Connected Layer
# hidden Layer
classifier.add(Dense(units=128,activation='relu'))
# Adding Second Fully Connected Layer
classifier.add(Dense(units=64,activation='relu'))
# output layer
classifier.add(Dense(units=1,activation='sigmoid'))


# 5 - Compile Model
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6 - Image Preprocessing
# Code Reference https://keras.io/preprocessing/image/
# Flow from Directory
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
classifier.save('Cat_Dog_Classfication.h5')