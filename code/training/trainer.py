import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import model as vae
import torch
from datetime import datetime
from preprocessing.dataset import dataset
import matplotlib.pyplot as plt


class trainer(object):
    """
    
    """
    def __init__(self, model, optimizer, train_loader, valid_loader, num_epochs: int = 50):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.train_losses = []
        self.valid_losses = []


    def run(self):
        self.train()
        self.save()

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        # Move model to GPU
        self.model.to(device)

        for epoch in range(self.num_epochs):
            starting_time = datetime.now()
            epoch_train_loss = 0.0
            epoch_valid_loss = 0.0

            # Training
            self.model.train()
            for batch_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, _ = data
                inputs = x.to(device)
                x_recon, mu, logvar = self.model(inputs)
                train_loss = self.model.loss_function(x, x_recon, mu, logvar)
                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss += train_loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(self.valid_loader):
                    x, _ = data
                    x_recon, mu, logvar = self.model(x)
                    valid_loss = self.model.loss_function(x, x_recon, mu, logvar)
                    epoch_valid_loss += valid_loss.item()

            epoch_train_loss /= len(self.train_loader)
            epoch_valid_loss /= len(self.valid_loader)

            self.train_losses.append(epoch_train_loss)
            self.valid_losses.append(epoch_valid_loss)

            time_elapsed = datetime.now() - starting_time
            print('Epoch: {}, Train loss: {:.4f}, Valid loss: {:.4f}, Time elapsed (hh:mm:ss.ms) {}'.format(
                    epoch, epoch_train_loss, epoch_valid_loss, time_elapsed
                ))

    
    def plot_losses(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.valid_losses, label='Validation loss')
        plt.legend()
        plt.savefig('docs/losses.pdf', format='pdf')
        # plt.show()
    
    def save(self):
        torch.save(self.model.state_dict(), 'vae.pt')


