import torch
from torch.utils.data import Dataset
import uproot
import numpy as np
import pandas as pd
import h5py
from numpy.lib.recfunctions import append_fields



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
        elif sg_files[0].endswith(".h5") and bkg_files[0].endswith(".h5"):
            print("Using h5 files")
            dtype = [('Track.PT', '<f4'), ('Track.Eta', '<f4'), ('Track.Phi', '<f4'), ('Track.D0', '<f4'), ('Track.DZ', '<f4')]
            data = np.array([], dtype=dtype)
            self.labels = np.array([])
            for idx,file_dir in enumerate(sg_files):
                d = self.get_data_from_h5(file_dir)
                self.labels = np.concatenate((self.labels,np.ones(d.shape[0])))
                data = np.append(data,d)
                print(f'Signal {idx}')
                    
            for idx,file_dir in enumerate(bkg_files):
                d = self.get_data_from_h5(file_dir)
                self.labels = np.concatenate((self.labels,np.zeros(d.shape[0])))
                data = np.append(data,d)
                print(f'Background {idx}')
            data = np.array([list(element) for element in data.tolist()])
        else:
            raise ValueError(f"data_file must be a .h5 or .csv file, not {data_file}")
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.data = -1 + 2 * (data - min_vals) / (max_vals - min_vals)
        self.num_features = len(self.data[0])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        print(self.data.shape)
        
    def get_data_from_h5(self, file_dir:str, bucket_name:str = 'cuda-programming-406720'):       
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        with h5py.File(file_contents, 'r') as f:
            dataset = f['Track']
            data = dataset[:100000]
        return data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # input_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample, sample
    
    def describe(self):
        print(f"Dataset has {len(self.data)} samples")