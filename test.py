import numpy as np

learning_rates_40 = np.concatenate([np.arange(0.0001, 0.0011, 0.0001), np.arange(0.002, 0.011, 0.001), np.arange(0.02, 0.11, 0.01)])
print(learning_rates_40)
learning_rates_80 = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1)])
print(learning_rates_80)
learning_rates_160 = np.logspace(-5, -1, num=20)
print(learning_rates_160)