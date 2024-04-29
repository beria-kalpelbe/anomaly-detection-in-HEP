import torch
from torch.utils.data import Dataset
import uproot
import numpy as np
import pandas as pd
import h5py
from numpy.lib.recfunctions import append_fields

# Before using, dataset object should be converted in datalader: 

class dataset(Dataset):
    """
    Dataset class for HEP data.

    Args:
        data_file (str): Path to the data file.

    Attributes:
        data (pd.DataFrame): Dataframe containing the data.
    """
    def __init__(self, data_file:str='',sg_files:str='', bkg_files:str='' ):
        if data_file.endswith(".csv"):
            # self.data = pd.read_csv(data_file)
            data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
            self.labels = data[:,-1]
            data = data[:,:-1]
        elif sg_files[0].endswith(".h5") and bkg_files[0].endswith(".h5"):
            for idx,file_dir in enumerate(sg_files):
                with h5py.File(file_dir, 'r') as file:
                    if idx==0:
                        d = file['Track'][:]
                    else:
                        d = np.concatenate((data,file['Track'][:]))
                        d = append_fields(d, 'label', np.zeros((d.shape[0])))
                data = np.concatenate((data,d.data))
                    
            for idx,file_dir in enumerate(bkg_files):
                with h5py.File(file_dir, 'r') as file:
                    d = file['Track'][:]
                    d = append_fields(d, 'label', np.ones((d.shape[0])))
                data = np.concatenate((data,d.data))

        elif data_file.endswith(".root"):
            tree = uproot.open(data_file)['tree']
            data = tree['Delphes']
            # Track level
            track_values_p = np.concatenate(data["Track"]["Track.P"].array())
            track_values_charge = np.concatenate(data["Track"]["Track.Charge"].array())
            track_values_pt = np.concatenate(data["Track"]["Track.PT"].array())
            track_values_eta = np.concatenate(data["Track"]["Track.Eta"].array())
            track_values_phi = np.concatenate(data["Track"]["Track.Phi"].array())
            track_values_ctgtheta = np.concatenate(data["Track"]["Track.CtgTheta"].array())
            track_values_d0 = np.concatenate(data["Track"]["Track.D0"].array())
            track_values_dz = np.concatenate(data["Track"]["Track.DZ"].array())
            # Hits values
            # hit_values_r = np.concatenate(data["Hits"]["Hits.r"].array())
            # hit_values_z = np.concatenate(data["Hits"]["Hits.z"].array())
            # hit_values_phi = np.concatenate(data["Hits"]["Hits.phi"].array())
            # hit_values_partIdx = np.concatenate(data["Hits"]["Hits.partIdx"].array())
            # hit_values_vxTruth = np.concatenate(data["Hits"]["Hits.vxTruth"].array())
            min_len = len(track_values_p)
            data_ = [
                track_values_p[:min_len],
                track_values_charge[:min_len],
                track_values_pt[:min_len],
                track_values_eta[:min_len],
                track_values_phi[:min_len],
                track_values_ctgtheta[:min_len],
                track_values_d0[:min_len],
                track_values_dz[:min_len],
                # hit_values_r[:min_len],
                # hit_values_z[:min_len],
                # hit_values_phi[:min_len],
                # hit_values_partIdx[:min_len],
                # hit_values_vxTruth[:min_len],
            ]
            data = np.array(data_)
        else:
            raise ValueError(f"data_file must be a .h5, .root or .csv file, not {data_file}")
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.data = -1 + 2 * (data - min_vals) / (max_vals - min_vals)
        self.num_features = len(self.data[0])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # input_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample, sample
    
    def describe(self):
        print(f"Dataset has {len(self.data)} samples")
