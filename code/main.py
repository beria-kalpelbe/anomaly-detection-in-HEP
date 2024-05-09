from preprocessing.dataset import dataset
from torch.utils.data import DataLoader
from training.trainer import trainer
from models.vae import VAE
from models.ode import ode
from evaluation.evaluator import evaluator
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
import torch
import os
import sys
from copy import copy
from torchsummary import summary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    if len(sys.argv) != 2:
        raise("Usage: python main.py model-name (vae, ode, bdt)")
    
    # data = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    # data_train = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    # data_valid = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')
    # data_test = dataset(data_file='/home/beria/Documents/anomaly-detection/data-clf.csv')

    
    data = dataset(data_file='QCD_LLP_samples/root-files/qcd_100k.root',
                    sg_files=['QCD_LLP_samples/h5-files/500GeV_n3_events_100k_1mm_pileup.h5', 'QCD_LLP_samples/h5-files/100GeV_n3_events_100k_1mm_pileup.h5'],
                    bkg_files=['QCD_LLP_samples/h5-files/QCD_multijet_events_200k_pileup.h5'])
  
    with open('code/utils/hyperparameters.json') as f:
        hyperparameters = json.load(f)

    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    lr = hyperparameters['LEARNING_RATE']
    num_epochs = hyperparameters['EPOCHS']
    latent_dim = hyperparameters['LATENT_DIM']
    
    train_loader = DataLoader(data.data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    valid_loader = DataLoader(data.data_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

    if sys.argv[1] == 'vae':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_vae = VAE(input_dim=data.num_features, latent_dim=latent_dim)
        model_vae = model_vae.to(device)
        
        summary(model_vae, input_size=(5,))
        
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

        evaluator_vae = evaluator(model=model_vae, test_data=data.data_test, labels=data.labels_test)
        evaluator_vae.run()
        evaluator_vae.describe_scores()
        evaluator_vae.event_embedding()
        
    if sys.argv[1] == 'bdt':
        bdt_model = DecisionTreeClassifier(random_state=0)
        bdt_model.fit(data.data_train, data.labels_train)
        
        bdt_evaluator = evaluator(model=bdt_model, test_data=data.data_test, labels=data.labels_test)
        bdt_evaluator.roc_curve()
        bdt_evaluator.confusion_matrix()
        bdt_evaluator.classification_report()
    
    if sys.argv[1] == 'ode':
        ode_model = ode(sg_data = data.sg_data, bkg_data=data.bkg_data)
        ode_model.run()
        ode_model.roc_curve(data)
        
        
           
        
        
    
    
if __name__ == '__main__':
    main()