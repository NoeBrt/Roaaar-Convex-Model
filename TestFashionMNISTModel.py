import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from FashionMNISTModel import FashionMNISTModel
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

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
            try:
                print(f"\nTesting with learning rate: {lr} and epochs: {epochs}")
                keras.backend.clear_session()  # Clear previous models from memory
                self.fashion_model.build_model()
                optimizer = optimizer_class(learning_rate=lr)  # Create a new optimizer instance with the current learning rate
                history = self.fashion_model.train_model(optimizer, epochs, verbose)
                for epoch in range(len(history.history['val_accuracy'])):
                    accuracy = history.history['val_accuracy'][epoch]
                    results.append((epoch + 1, lr, accuracy))
            except Exception as e:
                print(f"An error occurred with learning rate {lr}: {e}")
                continue
        return results

    def save_plot_results(self, results, epochs, optimizer_name):
        fig = plt.figure(figsize=(12, 6))

        # Extract data for plotting
        epochs_list = np.array([result[0] for result in results])
        learning_rates = np.array([result[1] for result in results])
        accuracies = np.array([result[2] for result in results])

        # Create a grid for plotting surface
        epochs_grid, lr_grid = np.meshgrid(np.unique(learning_rates), np.unique(epochs_list))
        accuracies_grid = np.zeros_like(epochs_grid, dtype=float)

        for i in range(epochs_grid.shape[0]):
            for j in range(epochs_grid.shape[1]):
                condition = (epochs_list == epochs_grid[i, j]) & (learning_rates == lr_grid[i, j])
                if np.any(condition):
                    accuracies_grid[i, j] = accuracies[condition][0]

        # Plot for the optimizer
        ax1 = fig.add_subplot(121, projection='3d')
        surface = ax1.plot_surface(epochs_grid, lr_grid, accuracies_grid, cmap='viridis')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Learning Rate')
        ax1.set_zlabel('Validation Accuracy')
        title = f"{optimizer_name}_FashionMNIST"
        ax1.set_title(title)
        fig.colorbar(surface, ax=ax1, label='Validation Accuracy')

        # Save the plot
        file_path = os.path.join(self.results_dir, f"{optimizer_name}-{epochs}.png")
        fig.savefig(file_path)
        plt.close(fig)

    def plot_and_show_results(self, all_results):
        for key, results in all_results.items():
            optimizer_name, epochs = key.split('_')
            epochs = int(epochs)
            self.save_plot_results(results, epochs, optimizer_name)

if __name__ == "__main__":
    test_model = TestFashionMNISTModel()
    
    optimizers = {
        "Lion": keras.optimizers.Adam,  # Replace with actual optimizer if available
        "Nadam": keras.optimizers.Nadam,
        "Adam": keras.optimizers.Adam,
        "RMSprop": keras.optimizers.RMSprop,
        "Adagrad": keras.optimizers.Adagrad,
        "AdamW": keras.optimizers.AdamW,
        "SGD": keras.optimizers.SGD
    }
    print("Optimizers to test:", optimizers.values())

    # Define learning rate ranges for each epoch configuration
    learning_rates_40 = np.concatenate([np.arange(0.0001, 0.0011, 0.0001), np.arange(0.002, 0.011, 0.001), np.arange(0.02, 0.11, 0.01)])
    learning_rates_80 = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1)])
    learning_rates_160 = np.logspace(-5, -1, num=20)  # Example learning rates for 160 epochs

    def process_optimizer(optimizer_name, optimizer_class, test_model, learning_rates, epochs):
        results = test_model.test_with_optimizer(optimizer_class, epochs, learning_rates)
        return optimizer_name, epochs, results

    # Create a dictionary to store results
    all_results = {}
    for optimizer_name, optimizer_class in optimizers.items():
        process_optimizer(optimizer_name, optimizer_class, test_model, learning_rates_40, 40)
    # Use ThreadPoolExecutor to parallelize the tasks
    '''
    with ThreadPoolExecutor() as executor:
        futures = []
        for optimizer_name, optimizer_class in optimizers.items():
            futures.append(executor.submit(process_optimizer, optimizer_name, optimizer_class, test_model, learning_rates_40, 40))
            futures.append(executor.submit(process_optimizer, optimizer_name, optimizer_class, test_model, learning_rates_80, 80))
            futures.append(executor.submit(process_optimizer, optimizer_name, optimizer_class, test_model, learning_rates_160, 160))

        for future in as_completed(futures):
            optimizer_name, epochs, results = future.result()
            all_results[f"{optimizer_name}_{epochs}"] = results
    '''

    csv_file_path = os.path.join('results', 'all_results.csv')
    os.makedirs('results', exist_ok=True)  # Ensure the results directory exists
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Optimizer', 'Epochs', 'Learning Rate', 'Validation Accuracy'])
        for key, value in all_results.items():
            optimizer_name, epochs = key.split('_')
            for result in value:
                epoch, learning_rate, accuracy = result
                writer.writerow([optimizer_name, epochs, learning_rate, accuracy])

    test_model.plot_and_show_results(all_results)
    print(f"Results saved to {csv_file_path}")
