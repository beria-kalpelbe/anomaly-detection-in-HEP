from preprocessing.dataset import dataset
from torch.utils.data import DataLoader
from training.trainer import trainer
from models.vae import model as vae
import torch.optim as optim
from sklearn.model_selection import train_test_split


def main():
    data = dataset(data_file='/home/beria/Documents/anomaly-detection/data.csv')
    data_train = dataset(data_file='/home/beria/Documents/anomaly-detection/data.csv')
    data_valid = dataset(data_file='/home/beria/Documents/anomaly-detection/data.csv')
    
    data_train.data, data_valid.data = train_test_split(data.data, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(data_train, batch_size=5, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=5, shuffle=True)

    model_vae = vae(input_dim=data.num_features, latent_dim=5)
    trainer_vae = trainer(model=model_vae, 
                          optimizer=optim.Adam(model_vae.parameters(), lr=1e-2), 
                          train_loader=train_loader,
                          valid_loader=valid_loader,
                          num_epochs=50)
    trainer_vae.run()
    trainer_vae.plot_losses()
    # model = vae()
    # trainer(model, dataloader)
    
    
if __name__ == '__main__':
    main()