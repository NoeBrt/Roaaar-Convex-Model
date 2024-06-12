import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

from IPython.display import Markdown
from importlib import reload
tf.test.is_gpu_available()


data = keras.datasets.fashion_mnist
cifar10_data = data . load_data ()

(train_images, train_labels),(test_images, test_labels) = cifar10_data

print("train_images shape : ",train_images.shape)

print("train_labels size", len(train_labels))

print("train_labels",train_labels)

print("test_images shape : ",test_images.shape)

print("test_labels",test_labels)

class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

index = 50
plt.figure()
plt.imshow(train_images[index])
plt.grid(False)
plt.title(class_names[train_labels[index]])
plt.show()

train_labels[index]


plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5, 5, i + 1)  
    plt.imshow(train_images[i]) 
    plt.title(class_names[int(train_labels[i])])  
    plt.axis('off') 

plt.show()


train_images = train_images / 255.0 
test_images = test_images / 255.0

train_images[0].shape



from keras import models
from keras import layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model




es = EarlyStopping(monitor="val_loss",patience=20)
mc = ModelCheckpoint("model_best.keras",save_best_only=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=200,batch_size=128,validation_split=0.2,validation_data=(test_images,test_labels),callbacks=[es,mc])