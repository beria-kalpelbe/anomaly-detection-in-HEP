from google.cloud import storage
from io import BytesIO
import h5py
import numpy as np

class dataset:
    class dataset:
        """
        A class for loading and accessing dataset for anomaly detection in High Energy Physics (HEP).
        """

        def __init__(self, bucket_name:str='cuda-programming-406720', file_path:str='QCD_LLP_samples/preprocessed_data.h5'):
            """
            Initialize the dataset object.

            Parameters:
            - bucket_name (str): The name of the bucket where the data file is stored.
            - file_path (str): The path to the data file within the bucket.
            """
            self.bucket_name = bucket_name
            self.file_path = file_path
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None
            self.load_data()
                    
        def load_data(self):
            """
            Load the data from the specified file.
            """
            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_path)
            file_contents = BytesIO(blob.download_as_string())
            with h5py.File(file_contents, 'r') as f:
                X_train = f['X_train'][:]
                y_train = f['y_train'][:]
                X_valid = f['X_valid'][:]
                y_valid = f['y_valid'][:]
                X_test = f['X_test'][:]
                y_test = f['y_test'][:]
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid = X_valid
            self.y_valid = y_valid
            self.X_test = X_test
            self.y_test = y_test
        
        def get_train_background(self, size=None):
            """
            Get the background samples from the training set.

            Returns:
            - ndarray: The background samples from the training set.
            """
            if size is None:
                return self.X_train[self.y_train == 0, :]
            else:
                return self.X_train[self.y_train == 0,:][:size, :]
        
        def get_train_signal(self, size=None):
            """
            Get the signal samples from the training set.

            Returns:
            - ndarray: The signal samples from the training set.
            """
            if size is None:
                return self.X_train[self.y_train == 1, :]
            else:
                return self.X_train[self.y_train == 1,:][:size, :]

        
        def get_valid_background(self, size=None):
            """
            Get the background samples from the validation set.

            Returns:
            - ndarray: The background samples from the validation set.
            """
            if size is None:
                return self.X_valid[self.y_valid == 0, :]
            else:
                return self.X_valid[self.y_valid == 0,:][:size, :]
        
        def get_valid_signal(self, size=None):
            """
            Get the signal samples from the validation set.

            Returns:
            - ndarray: The signal samples from the validation set.
            """
            if size is None:
                return self.X_valid[self.y_valid == 1, :]
            else:
                return self.X_valid[self.y_valid == 1,:][:size, :]
        
        def get_test_background(self, size=None):
            """
            Get the background samples from the test set.

            Returns:
            - ndarray: The background samples from the test set.
            """
            if size is None:
                return self.X_test[self.y_test == 0, :]
            else:    
                return self.X_test[self.y_test == 0,:][:size, :]
        
        def get_test_signal(self, size=None):
            """
            Get the signal samples from the test set.

            Returns:
            - ndarray: The signal samples from the test set.
            """
            if size is None:
                return self.X_test[self.y_test == 1, :]
            else:
                return self.X_test[self.y_test == 1,:][:size, :]
    