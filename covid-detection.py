
import os
import numpy as np
import cv2
from numpy import random
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from matplotlib import pyplot
import matplotlib.patheffects as path_effects

input_path = 'chest_xray/'
#defined some constants for later usage
img_dims = 160
epochs = 3
batch_size = 32
class_names = ['/NORMAL/', '/COVID-19/']


#Fitting the CNN to the images
def process_data(img_dims, batch_size):
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, shear_range=0.2, vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        directory=input_path + 'train',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    val_gen = val_datagen.flow_from_directory(
        directory=input_path + 'val',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []
    label = 0
    for cond in ['/NORMAL/', '/COVID-19/']:
        for img in tqdm(os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path + 'test' + cond + img)
            img = cv2.resize(img, (img_dims, img_dims))
            if len(img.shape) == 1:
                img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            if cond == '/NORMAL/':
                label = 0
            elif cond == '/COVID-19/':
                label = 1
            test_data.append(img)
            test_labels.append(label)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_gen, val_gen, test_data, test_labels

train_gen, val_gen, test_data, test_labels = process_data(img_dims, batch_size)

#Initialising the CNN
model = models.Sequential()

model.add(layers.Conv2D(16,(3,3),activation = 'relu', padding='same',input_shape=(img_dims,img_dims,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(256,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.3))

model.add(layers.Flatten())
# Creating 1 Dense Layer
model.add(layers.Dense(units=128,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=1, activation='sigmoid'))



#Compiling the CNN
model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

#model.load_weights('best_weights.hdf5')

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=2, mode='max')
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

hist = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[checkpoint, lr_reduce])

test_loss, test_acc = model.evaluate(test_data, test_labels)

##Predictions
predictions = model.predict(test_data)
predictions = np.array(predictions).reshape(-1)

##Accuracy
print("Untrained model, accuracy: {:5.2f}%".format(100 * test_acc))

acc = accuracy_score(test_labels, np.round(predictions))*100
cm = confusion_matrix(test_labels, np.round(predictions))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

#Visualize the accuracy plots and the model loss
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
   ax[i].plot(hist.history[met])
   ax[i].plot(hist.history['val_' + met])
   ax[i].set_title('Model {}'.format(met))
   ax[i].set_xlabel('epochs')
   ax[i].set_ylabel(met)
   ax[i].legend(['train', 'val'])
plt.show()
########################################################