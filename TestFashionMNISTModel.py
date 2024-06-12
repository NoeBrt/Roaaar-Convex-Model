import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from FashionMNISTModel import FashionMNISTModel
class TestFashionMNISTModel:
    def __init__(self):
        self.fashion_model = FashionMNISTModel()
        self.fashion_model.load_data()
        self.fashion_model.preprocess_data()
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def test_with_optimizer(self, optimizer_class, epochs, learning_rates, verbose=1):
        results = []
        for lr in learning_rates:
            print(f"\nTesting with learning rate: {lr} and epochs: {epochs}")
            keras.backend.clear_session()  # Clear previous models from memory
            self.fashion_model.build_model()
            optimizer = optimizer_class(learning_rate=lr)  # Create a new optimizer instance with the current learning rate
            history = self.fashion_model.train_model(optimizer, epochs, verbose)
            for epoch in range(len(history.history['val_accuracy'])):
                accuracy = history.history['val_accuracy'][epoch]
                results.append((epoch + 1, lr, accuracy))
        return results

    def plot_results(self, results, epochs, optimizer_name):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        epochs_list = [result[0] for result in results]
        learning_rates = [result[1] for result in results]
        accuracies = [result[2] for result in results]

        scatter = ax.scatter(epochs_list, learning_rates, accuracies, c=accuracies, cmap='viridis')

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_zlabel('Validation Accuracy')
        title = f"{optimizer_name}_FashionMNIST"
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, label='Validation Accuracy')

        # Save the plot
        file_path = os.path.join(self.results_dir, f"{optimizer_name}-{epochs}.png")
        plt.savefig(file_path)
        plt.show()

if __name__ == "__main__":
    test_model = TestFashionMNISTModel()
    
    optimizers = {
        "Adam": keras.optimizers.Adam,
        "RMSprop": keras.optimizers.RMSprop,
        "Adagrad": keras.optimizers.Adagrad
    }

    # Define learning rate ranges for each epoch configuration
    learning_rates_40 = np.concatenate([np.arange(0.0001, 0.0011, 0.0001), np.arange(0.002, 0.011, 0.001), np.arange(0.02, 0.11, 0.01)])
    learning_rates_80 = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1)])
    learning_rates_160 = np.logspace(-5, -1, num=20)  # Example learning rates for 160 epochs

    for optimizer_name, optimizer_class in optimizers.items():
        results_40 = test_model.test_with_optimizer(optimizer_class, 40, learning_rates_40)
        results_80 = test_model.test_with_optimizer(optimizer_class, 80, learning_rates_80)
        results_160 = test_model.test_with_optimizer(optimizer_class, 160, learning_rates_160)

        test_model.plot_results(results_40, 40, f"{optimizer_name}_40")
        test_model.plot_results(results_80, 80, f"{optimizer_name}_80")
        test_model.plot_results(results_160, 160, f"{optimizer_name}_160")