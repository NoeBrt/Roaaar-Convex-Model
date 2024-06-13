import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import keras.backend as K
class FashionMNISTModel:
    def __init__(self):
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.model = None
        self.data = None
        tf.config.list_physical_devices('GPU')

    def load_data(self):
        print("Loading data...")
        self.data = keras.datasets.fashion_mnist
        print("Data loaded successfully!")

    def preprocess_data(self):
        print("Preprocessing data...")
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.data.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        
        self.train_images = np.expand_dims(self.train_images, -1)
        self.test_images = np.expand_dims(self.test_images, -1)
        print("Data preprocessed successfully!")

    def build_model(self):
        print("Building model...")
        from keras import models, layers, regularizers

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        self.model.summary()
        print("Model built successfully!")

    def train_model(self, optimizer, epochs, verbose=1, best_model_path="models/model_best", batch_size=128, validation_split=0.2, patience=20):
        from keras.callbacks import ModelCheckpoint, EarlyStopping

        print("Training model...")
        es = EarlyStopping(monitor="val_loss", patience=patience)
        mc = ModelCheckpoint(f"{best_model_path}-{optimizer.__class__.__name__}-ep{epochs}-lr{optimizer.learning_rate.numpy()}.keras", save_best_only=True)
        
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = self.model.fit(
            self.train_images, self.train_labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            validation_data=(self.test_images, self.test_labels), 
            callbacks=[es, mc], 
            verbose=verbose
        )
        
        print("Model trained successfully!")
        return history

    def evaluate_model(self):
        print("Evaluating model...")
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print('\nTest accuracy:', test_acc)
        return test_acc


