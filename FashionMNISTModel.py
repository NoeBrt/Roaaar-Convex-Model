import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class FashionMNISTModel:
    def __init__(self):
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        self.model = None
        self.data = None

    def load_data(self):
        self.data = keras.datasets.fashion_mnist
        
    def preprocess_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.data.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        
        self.train_images = np.expand_dims(self.train_images, -1)
        self.test_images = np.expand_dims(self.test_images, -1)

    def build_model(self):
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

    def train_model(self, optimizer, epochs, verbose=1, best_model_path="models/model_best", batch_size=128, validation_split=0.2, patience=20):
        from keras.callbacks import ModelCheckpoint, EarlyStopping

        es = EarlyStopping(monitor="val_loss", patience=patience)
        mc = ModelCheckpoint(f"{best_model_path}-{optimizer.__class__.__name__}-ep{epochs}.keras", save_best_only=True)
        
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
        
        return history


    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print('\nTest accuracy:', test_acc)
        return test_acc

    def visualize_sample(self, index):
        plt.figure()
        plt.imshow(self.train_images[index].reshape(28, 28), cmap='gray')
        plt.grid(False)
        plt.title(self.class_names[self.train_labels[index]])
        plt.show()

    def visualize_samples(self):
        plt.figure(figsize=(20, 20))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.train_images[i].reshape(28, 28), cmap='gray')
            plt.title(self.class_names[int(self.train_labels[i])])
            plt.axis('off')
        plt.show()
