import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from FashionMNISTModel import FashionMNISTModel
import concurrent.futures
import csv 

class TestFashionMNISTModel:
    def __init__(self):
        self.fashion_model = FashionMNISTModel()
        self.fashion_model.load_data()
        self.fashion_model.preprocess_data()
        self.results_dir = "results"
        self.test_accuracy = {}
        # test gpu
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)

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

    def save_plot_results(self, results, epochs, optimizer_name):
        fig = plt.figure(figsize=(12, 6))

        # Extract data for plotting
        epochs_list = np.array([result[0] for result in results])
        learning_rates = np.array([result[1] for result in results])
        accuracies = np.array([result[2] for result in results])

        # Sort learning rates in descending order
        sorted_learning_rates = np.sort(np.unique(learning_rates))

        # Create a grid for plotting surface
        epochs_grid, lr_grid = np.meshgrid(np.unique(epochs_list), sorted_learning_rates)
        accuracies_grid = np.zeros_like(epochs_grid, dtype=float)

        for i in range(epochs_grid.shape[0]):
            for j in range(epochs_grid.shape[1]):
                condition = (epochs_list == epochs_grid[i, j]) & (learning_rates == lr_grid[i, j])
                if np.any(condition):
                    accuracies_grid[i, j] = accuracies[condition][0]

        # Plot for the optimizer
        ax1 = fig.add_subplot(111, projection='3d')
        surface = ax1.plot_surface(epochs_grid, lr_grid, accuracies_grid, cmap='viridis')
        ax1.set_xlabel('Epochs')
        #invert y axis
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_ylabel('Learning Rate')
        ax1.set_zlabel('Validation Accuracy')
        title = f"{optimizer_name}_FashionMNIST"
        ax1.set_title(title)

        # Save the plot
        file_path = os.path.join(self.results_dir, f"{optimizer_name}-{epochs}.png")
        plt.tight_layout()
        fig.savefig(file_path)
        plt.close(fig)


    def plot_and_show_results(self, all_results):
        for key, results in all_results.items():
            optimizer_name, epochs = key.split('_')
            epochs = int(epochs)
            self.save_plot_results(results, epochs, optimizer_name)

    def run_test_with_optimizer(self, optimizer_name, optimizer_class, epochs, learning_rates):
        model = FashionMNISTModel()
        model.load_data()
        model.preprocess_data()
        results = []
        for lr in learning_rates:
            print(f"\nTesting with learning rate: {lr} and epochs: {epochs}")
            keras.backend.clear_session()  # Clear previous models from memory
            model.build_model()
            optimizer = optimizer_class(learning_rate=lr)  # Create a new optimizer instance with the current learning rate
            history = model.train_model(optimizer, epochs, 1)  # Verbose is set to 1
            for epoch in range(len(history.history['val_accuracy'])):
                accuracy = history.history['val_accuracy'][epoch]
                results.append((epoch + 1, lr, accuracy))
            self.test_accuracy[f"{optimizer_name}-{lr}"] = model.evaluate_model()
        return optimizer_name, epochs, results

    def save_results_to_csv(self, all_results):
        csv_file = os.path.join(self.results_dir, "all_results.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Optimizer", "Epoch", "Learning Rate", "Accuracy"])
            for key, results in all_results.items():
                optimizer_name, epochs = key.split('_')
                for result in results:
                    writer.writerow([optimizer_name, result[0], result[1], result[2]])

    def save_test_accuracy_to_csv(self, test_accuracy):
        csv_file = os.path.join(self.results_dir, "test_accuracy.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Optimizer", "Accuracy"])
            for key, value in test_accuracy.items():
                writer.writerow([key, value])

if __name__ == "__main__":
    test_model = TestFashionMNISTModel()
    
    optimizers = {
        "Lion": keras.optimizers.Lion,
        "Nadam": keras.optimizers.Nadam,
        "Adam": keras.optimizers.Adam,
        "RMSprop": keras.optimizers.RMSprop,
        "Adagrad": keras.optimizers.Adagrad,
        "AdamW": keras.optimizers.AdamW,
        "SGD": keras.optimizers.SGD
    }

    # Define learning rate ranges for each epoch configuration
    learning_rates_40 = np.concatenate([np.arange(0.0001, 0.0011, 0.0001), np.arange(0.002, 0.011, 0.001), np.arange(0.02, 0.11, 0.01)])
    learning_rates_80 = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1)])
    learning_rates_160 = np.logspace(-5, -1, num=20)  # Example learning rates for 160 epochs

    def process_optimizer(optimizer_name, optimizer_class, learning_rates, epochs):
        return test_model.run_test_with_optimizer(optimizer_name, optimizer_class, epochs, learning_rates)

    # Create a dictionary to store results
    all_results = {}

    for optimizer_name, optimizer_class in optimizers.items():
        epochs = 40
        _, _, results_40 = process_optimizer(optimizer_name, optimizer_class, learning_rates_40, epochs)
        all_results[f"{optimizer_name}_40"] = results_40
        test_model.save_plot_results(results_40, epochs, optimizer_name)

    test_model.plot_and_show_results(all_results)
    test_model.save_results_to_csv(all_results)
    test_model.save_test_accuracy_to_csv(test_model.test_accuracy)
