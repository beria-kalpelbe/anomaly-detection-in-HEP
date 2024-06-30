from google.cloud import storage
from io import BytesIO
import h5py
import numpy as np
import uproot
import torch
from itertools import chain
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker



class preprocess:
    """
    Class for preprocessing data for anomaly detection in High Energy Physics (HEP).
    
    Args:
        bucket_name (str): Name of the bucket where the data is stored.
        features (list): List of features to be used for preprocessing.
        signal_files (list): List of signal file paths.
        background_file (str): File path of the background data.
        data_size (int): Total size of the data to be used.
    """
    
    def __init__(self, bucket_name:str = 'cuda-programming-406720', 
                 features:list = ['Track.PT', 'Track.Eta', 'Track.Phi', 'Track.D0', 'Track.DZ'],
                 signal_files:list = [
                     'QCD_LLP_samples/h5-files/500GeV_n3_events_100k_1mm_pileup.h5', 
                     'QCD_LLP_samples/h5-files/100GeV_n3_events_100k_1mm_pileup.h5'
                     ],
                 background_file:str = 'QCD_LLP_samples/root-files/qcd_100k.root',
                 data_size:int = 12_500_000):
        self.features = features
        self.bucket_name = bucket_name
        self._read_data(signal_files, background_file)
        self.data_size = data_size
    
    def _get_data_from_root(self, file_dir:str):
        """
        Helper method to read data from a ROOT file.
        
        Args:
            file_dir (str): File path of the ROOT file.
        
        Returns:
            data: Data read from the ROOT file.
        """
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        tree = uproot.open(file_contents)
        data = tree['Delphes']
        return data
    
    def _get_data_from_h5(self, file_dir:str):
        """
        Helper method to read data from an HDF5 file.
        
        Args:
            file_dir (str): File path of the HDF5 file.
        
        Returns:
            d: Data read from the HDF5 file.
        """
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.blob(file_dir)
        file_contents = BytesIO(blob.download_as_string())
        with h5py.File(file_contents, 'r') as f:
            dataset = f['Track']
            d = dataset[:]
        d = torch.tensor([list(d[i]) for i in range(d.shape[0])])
        return d
    
    def select_data_in_root_data(self, data):
        """
        Selects the required data from the ROOT data.
        
        Args:
            data: Data read from the ROOT file.
        
        Returns:
            selected_data: Selected data from the ROOT data.
        """
        selected_data = []
        for i in range(len(self.features)):
            d = data['Track'][self.features[i]].array().tolist()
            d = list(chain.from_iterable(d))
            selected_data.append(d)
        selected_data = torch.tensor(selected_data).T
        return selected_data
    
    def _read_background_data(self, file_dir:str, features:list):
        """
        Reads the background data from the given file.
        
        Args:
            file_dir (str): File path of the background data.
            features (list): List of features to be used for preprocessing.
        
        Returns:
            data: Preprocessed background data.
        """
        if file_dir.endswith('.root'):
            data = self._get_data_from_root(file_dir)
            data = self.select_data_in_root_data(data)
        elif file_dir.endswith('.h5'):
            data = self._get_data_from_h5(file_dir)
        else:
            raise ValueError("Invalid file format. Only .root and .h5 files are supported.")
        data = torch.concatenate((data, torch.tensor([[0]]*data.shape[0])), axis=1)
        return data
    
    
    def _read_signal_data(self, file_dirs:list[str]):
        """
        Reads the signal data from the given files.
        
        Args:
            file_dirs (list): List of file paths of the signal data.
        
        Returns:
            concatenated_data: Preprocessed concatenated signal data.
        """
        concatenated_data = torch.tensor([])
        for file_dir in file_dirs:
            if file_dir.endswith('.root'):
                data = self._get_data_from_root(file_dir)
                data = self.select_data_in_root_data(data)
            elif file_dir.endswith('.h5'):
                data = self._get_data_from_h5(file_dir)
            else:
                raise ValueError("Invalid file format. Only .root and .h5 files are supported.")
            concatenated_data = torch.concatenate((concatenated_data, data))
        # concatenated_data = torch.tensor(concatenated_data)
        concatenated_data = torch.concatenate((concatenated_data, torch.tensor([[1]]*concatenated_data.shape[0])), axis=1)
        concatenated_data[:, 0] = concatenated_data[:, 0] * 1000 # convert GeV to MeV
        return concatenated_data
    
    
    def _read_data(self, signal_files:list[str], background_file:str):
        """
        Reads the signal and background data.
        
        Args:
            signal_files (list): List of file paths of the signal data.
            background_file (str): File path of the background data.
        """
        print('reading background dataset......')
        self.background_data = self._read_background_data(background_file, self.features)
        print('reading signal dataset......')
        self.signal_data = self._read_signal_data(signal_files)
        
    
    def print_desc_stats(self) -> None:
        """
        Prints the descriptive statistics of the data.
        """
        data = torch.cat((self.signal_data, self.background_data), dim=0)
        for i in range(len(self.features)):
            print('+------------------------------------------------------------------------------------+')
            feature_values = data[:, i]
            feature_mean = torch.mean(feature_values)
            feature_std = torch.std(feature_values)
            print(f"Feature: {self.features[i]}")
            print(f"Global - Range: [{torch.min(feature_values)}, {torch.max(feature_values)}]")
            print(f"Global - Mean: {feature_mean}")
            print(f"Global - Std: {feature_std}")
            
            signal_values = data[data[:, -1] == 1, i]
            signal_mean = torch.mean(signal_values)
            signal_std = torch.std(signal_values)
            print(f"Signal - Range: [{torch.min(signal_values)}, {torch.max(signal_values)}]")
            print(f"Signal - Mean: {signal_mean}")
            print(f"Signal - Std: {signal_std}")
            
            background_values = data[data[:, -1] == 0, i]
            background_mean = torch.mean(background_values)
            background_std = torch.std(background_values)
            print(f"Background - Range: [{torch.min(background_values)}, {torch.max(background_values)}]")
            print(f"Background - Mean: {background_mean}")
            print(f"Background - Std: {background_std}")

    def plot_hists(self, set: str, log_scale: bool = True, size_of_data_to_plot: int = 5000, savefig: str = None):
        """
        Plots histograms for the specified dataset.

        Parameters:
        - set (str): The dataset to plot histograms for. Supported values are 'signal', 'background', and 'all'.
        - log_scale (bool): Whether to use a logarithmic scale for the y-axis. Default is True.
        - size_of_data_to_plot (int): The number of data points to plot. Default is 5000.
        - savefig (str): The file path to save the plot as an image. If None, the plot will be displayed on the screen. Default is None.
        """

        xlabels = [r'$p_T$', r'$\eta$', r'$\phi$', r'$d_0$', r'$d_z$']
        if set == 'signal':
            data = self.signal_data[:size_of_data_to_plot, :]
        elif set == 'background':
            data = self.background_data[:size_of_data_to_plot, :]
        elif set == 'all':
            data = self.signal_data[:size_of_data_to_plot // 2, :]
            data2 = self.background_data[:size_of_data_to_plot // 2, :]
        else:
            raise ValueError("Invalid set. Only 'signal', 'background', and 'all' are supported.")

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(len(self.features)):
            if set == 'all':
                sns.histplot(data[:, i], ax=axes[i], bins=100, element="step", fill=False, stat="density", log_scale=log_scale, label='Signal')
                sns.histplot(data2[:, i], ax=axes[i], bins=100, element="step", fill=False, stat="density", log_scale=log_scale, label='Background')
                axes[0].legend(fontsize=8)
            else:
                sns.histplot(data[:, i], ax=axes[i], bins=100, element="step", fill=False, stat="density", log_scale=log_scale)
            axes[i].yaxis.set_major_formatter(ticker.ScalarFormatter('%.1f'))
            axes[i].set_xlabel(xlabels[i], fontsize=10)

        plt.subplots_adjust(wspace=0.4, hspace=0.2)
        if savefig is not None:
            plt.savefig(savefig)
        fig.show()
    
    

    def _split_data(self, test_size:float=0.2, random_state:int=45):
        """
        Splits the data into train, validation, and test sets.
        
        Args:
            test_size (float): Proportion of the data to be used for testing.
            random_state (int): Random seed for reproducibility.
        """
        data = torch.cat((self.signal_data[:(self.data_size//2), :], self.background_data[:(self.data_size//2), :]), dim=0)
        np.random.seed(random_state)
        np.random.shuffle(data)
        X = data[:,:5]
        y = data[:,-1]
        self.X_train, X_remain, self.y_train, y_remain  = train_test_split(
            X,y, test_size=test_size, random_state=random_state
            )
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_remain, y_remain, test_size=0.5, random_state=random_state
            )
    
    def save_data(self, output_file):
        """
        Saves the preprocessed data to an HDF5 file.
        
        Args:
            output_file (str): File path to save the data.
        """
        self._split_data()
        with h5py.File(output_file, 'w') as f:
            dset = f.create_dataset('X_train', data=self.X_train, dtype='f')
            dset = f.create_dataset('y_train', data=self.y_train, dtype='f')
            dset = f.create_dataset('X_valid', data=self.X_valid, dtype='f')
            dset = f.create_dataset('y_valid', data=self.y_valid, dtype='f')
            dset = f.create_dataset('X_test', data=self.X_test, dtype='f')
            dset = f.create_dataset('y_test', data=self.y_test, dtype='f')
        print(f"Data has been written to {output_file}")
