from sklearn.model_selection import train_test_split
from google.cloud import storage
from io import BytesIO
import uproot
from itertools import chain
import numpy as np
import torch
import h5py
from array import array
import os



# Signal dataset
def get_data_from_h5(file_dir:str, bucket_name:str = 'cuda-programming-406720'):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        with h5py.File(file_contents, 'r') as f:
            dataset = f['Track'] 
            d = dataset[:]
        return d
print('reading signal data .....')
data_sig1 = get_data_from_h5('QCD_LLP_samples/h5-files/500GeV_n3_events_100k_1mm_pileup.h5')
data_sig2 = get_data_from_h5('QCD_LLP_samples/h5-files/100GeV_n3_events_100k_1mm_pileup.h5')
data_sig = np.concatenate((data_sig1,data_sig2))
data_sig = torch.tensor([list(data_sig[i]) for i in range(data_sig.shape[0])])

data_sig = torch.concatenate((data_sig, torch.tensor([[1]]*data_sig.shape[0])), axis=1)
print('Shape of signal dataset:', data_sig.shape)

# Background dataset
def get_data_from_root(file_dir:str, bucket_name:str = 'cuda-programming-406720'):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        tree = uproot.open(file_contents)
        data = tree['Delphes']
        return data
print('reading background data .....')
data_file = 'QCD_LLP_samples/root-files/qcd_100k.root'
data = get_data_from_root(data_file)
features = ['Track.PT', 'Track.Eta', 'Track.Phi', 'Track.D0', 'Track.DZ']

data_bkg = []
for i in range(5):
  d = data['Track'][features[i]].array().tolist()
  d = list(chain.from_iterable(d))
  data_bkg.append(d)
data_bkg = torch.tensor(data_bkg).T
data_bkg = torch.concatenate((data_bkg, torch.tensor([[0]]*data_bkg.shape[0])), axis=1)
print('Shape of background dataset:', data_bkg.shape)


print('Split the dataset')
data = torch.concatenate((data_bkg, data_sig))
print(data[0])
data_train, remain_data = train_test_split(data, test_size=0.2, random_state=45) 
data_valid, data_test = train_test_split(remain_data, test_size=0.5, random_state=45)
print(data_test[0])
# export the dataset cleaned
print('exporting the dataset .....')
def export_data_to_h5(train, test, valid, output_file):
    # temp_file = 'temp.h5'
    with h5py.File(output_file, 'w') as f:
      # Create a dataset in the file
      dset = f.create_dataset('train', train, dtype='f')
      dset = f.create_dataset('test', test, dtype='f')
      dset = f.create_dataset('valid', valid, dtype='f')
    print(f"Data has been written to {output_file}")


export_data_to_h5(train=data_train,
                  test=data_test,
                  valid=data_valid,
                  output_file='preprocessed_data.h5')
