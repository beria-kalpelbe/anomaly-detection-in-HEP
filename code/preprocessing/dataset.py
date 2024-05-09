import torch
from torch.utils.data import Dataset
import uproot
import numpy as np
import pandas as pd
import h5py
from numpy.lib.recfunctions import append_fields
from sklearn.model_selection import train_test_split


import h5py
from google.cloud import storage
from io import BytesIO




# Before using, dataset object should be converted in datalader: 

class dataset(Dataset):
    """
    Dataset class for HEP data.
    Args:
        data_file (str): Path to the data file.
        sg_files (str): Path to the signal files.
        bkg_files (str): Path to the background files.
    Attributes:
        data (pd.DataFrame): Dataframe containing the data.
    """
    def __init__(self, data_file:str='',sg_files:list[str]=[''], bkg_files:list[str]=['']):
        if data_file.endswith(".csv"):
            # self.data = pd.read_csv(data_file)
            data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
            self.labels = data[:,-1]
            data = data[:,:-1]
        if sg_files[0].endswith(".h5") and bkg_files[0].endswith(".h5"):
            print("Using h5 files")
            dtype = [('Track.PT', '<f4'), ('Track.Eta', '<f4'), ('Track.Phi', '<f4'), ('Track.D0', '<f4'), ('Track.DZ', '<f4')]
            data = np.array([], dtype=dtype)
            self.labels = np.array([])
            for idx,file_dir in enumerate(sg_files):
                d = self.get_data_from_h5(file_dir)
                self.labels = np.concatenate((self.labels,np.ones(d.shape[0])))
                data = np.append(data,d)
                print(f'Signal file: {file_dir}')
            self.sg_data = np.array([list(element) for element in data.tolist()])
            for idx,file_dir in enumerate(bkg_files):
                d = self.get_data_from_h5(file_dir)
                self.labels = np.concatenate((self.labels,np.zeros(d.shape[0])))
                data = np.append(data,d)
                print(f'Background file: {file_dir}')
            self.bkg_data = d
            data = np.array([list(element) for element in data.tolist()])
            self.test_data = data
        if data_file[0].endswith(".root"):
            print("Using root files")
            data_train = self.get_data_from_root(data_file)
            features = ['Track.PT', 'Track.Eta', 'Track.Phi', 'Track.D0', 'Track.DZ']
            data_ = []
            for i in range(5):
                d = data_train['Track'][features[i]].array().tolist()
                data_.append(d)
            data_ = np.array(data_)
            del data_train
            self.data_train = data_
        else:
            raise ValueError(f"data_file must be a .h5 or .csv file, not {data_file}")
        self.data_train = self.normalize(self.data_train)
        self.data_train = torch.tensor(self.data_train, dtype=torch.float32)
        self.data_test = self.normalize(self.data_test)
        self.split_data()
        
    def normalize(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        data = -1 + 2 * (data - min_vals) / (max_vals - min_vals)
        return data
        
    def get_data_from_h5(self, file_dir:str, bucket_name:str = 'cuda-programming-406720'):       
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        with h5py.File(file_contents, 'r') as f:
            dataset = f['Track']
            data = dataset[:1000]
        return data
    
    def get_data_from_root(self, file_dir:str, bucket_name:str = 'cuda-programming-406720'):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        tree = uproot.open(file_contents)
        data = tree['Delphes']
        return data
    
    def split_data(self):
        indices = np.array(range(self.data_train.shape[0]))
        np.random.seed(39)
        np.random.shuffle(indices)
        train_size = int(0.8*self.data_train.shape[0])
        valid_size  = int(0.2*self.data_train.shape[0])
        indices_train = indices[:train_size]
        indices_valid = indices[(train_size+1):(train_size + valid_size)]
        # indices_test = indices[(train_size + valid_size+1):]
        self.data_train = self.data_train[indices_train,:]
        # self.labels_train = self.labels[indices_train]
        self.data_valid = self.data_train[indices_valid,:]
        # self.labels_valid = self.labels[indices_valid]
        # self.data_test = self.data_train[indices_test,:]
        # self.labels_test = self.labels[indices_test]
        
        # self.data_train, data, self.train_labels, labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
        # self.data_valid, self.data_test, self.valid_labels, self.test_labels = train_test_split(data, labels, test_size=0.5, random_state=39)
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # input_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample, sample
    
    def describe(self):
        print(f"Dataset has {len(self.data)} samples")