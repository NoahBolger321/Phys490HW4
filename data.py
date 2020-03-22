import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class MnistData():
    def __init__(self, test_size):
        """
        :param test_size: specifies the size of the test set (decimal)

        - reads csv data into data frame and splits into data and labels
        - uses scikit learn train_test_split to split into train and test (n=3000) sets
        """
        mnist_df = pd.read_csv('data/even_mnist.csv', header=None, names=['data'])

        mnist_df['labels'] = pd.to_numeric(mnist_df['data'].str[-1])
        mnist_df['data'] = mnist_df['data'].apply(lambda d: np.array(list(map(float, d.split()[:-1])), dtype=np.float32))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                                                                mnist_df['data'],
                                                                mnist_df['labels'],
                                                                test_size=test_size)
