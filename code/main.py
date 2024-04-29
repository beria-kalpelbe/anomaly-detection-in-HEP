from preprocessing.dataset import dataset
from torch.utils.data import DataLoader
from training.trainer import trainer
from models.vae import model as vae, VAE
from evaluation.evaluator import evaluator
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
import torch
import os


def main():
    data = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    data_train = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    data_valid = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    data_test = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    
    # data = dataset(sg_files=['/home/beria/Documents/anomaly-detection/500GeV_n3_events_100k_1mm_pileup.h5',
    #                         '/home/beria/Documents/anomaly-detection/100GeV_n3_events_100k_1mm_pileup.h5'],
    #               bkg_files=['/home/beria/Documents/anomaly-detection/QCD_multijet_events_200k_pileup.h5'])
    # data_train = data
    # data_valid = data
    # data_test = data
    data_train.data, d, data_train.labels, labels = train_test_split(data.data, data.labels, test_size=0.2, random_state=42)
    data_valid.data, data_test.data, data_valid.labels, data_test.labels = train_test_split(d, labels, test_size=0.5, random_state=39)
  
    with open('code/utils/hyperparameters.json') as f:
        hyperparameters = json.load(f)

    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    lr = hyperparameters['LEARNING_RATE']
    num_epochs = hyperparameters['EPOCHS']
    latent_dim = hyperparameters['LATENT_DIM']
    
    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=True)
    model_vae = VAE(input_dim=data.num_features, latent_dim=latent_dim)
    
    model_path = 'vae.pt'
    if os.path.exists(model_path):
        print('A model saved vae.pt have been found and will be loaded')
        model_vae.load_state_dict(torch.load(model_path))
    else:
        trainer_vae = trainer(model=model_vae, 
                            optimizer=optim.Adam(model_vae.parameters(), lr=lr), 
                            train_loader=train_loader,
                            valid_loader=valid_loader,
                            num_epochs=num_epochs)
        trainer_vae.run()
        trainer_vae.plot_losses()

    evaluator_vae = evaluator(model=model_vae, test_data=data_test)
    evaluator_vae.run()
    evaluator_vae.describe_scores()
    
    
if __name__ == '__main__':
    main()