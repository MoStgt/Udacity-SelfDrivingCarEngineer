import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv')  as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []


for line in lines:
    measurement_center = float(line[3])
    
    correction = 0.2
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction
    
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image_center = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
    
    
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image_left = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)    
    
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image_right = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB) 
    
    images.extend((image_center, image_left, image_right))
    measurements.extend((measurement_center, measurement_left, measurement_right))

"""
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1)
"""    
    
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))           
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(64, (3, 3), activation="elu"))
model.add(Convolution2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) 


model.compile(loss='mse',optimizer='adam')


model.fit( X_train, y_train, validation_split=0.2, verbose=1, nb_epoch = 3)


model.save('model.h5')

print('Model Saved!')