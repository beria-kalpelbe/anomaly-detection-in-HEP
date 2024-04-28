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
        for epoch in range(self.num_epochs):
            starting_time = datetime.now()
            for batch_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, _ = data
                x_recon, mu, logvar = self.model(x)
                train_loss = self.model.loss_function(x, x_recon, mu, logvar)
                train_loss.backward()
                self.optimizer.step()
            self.train_losses.append(train_loss.item())
            with torch.no_grad():
                for batch_idx, data in enumerate(self.valid_loader):
                    x, _ = data
                    x_recon, mu, logvar = self.model(x)
                    valid_loss = self.model.loss_function(x, x_recon, mu, logvar)
            self.valid_losses.append(valid_loss.item())
            time_elapsed = datetime.now() - starting_time
            print('Epoch: {}, Batch: {}, Train loss: {:.4f}, Valid loss: {:.4f}, Time elapsed (hh:mm:ss.ms) {}'.format(
                    epoch, batch_idx, train_loss.item(), valid_loss.item(), time_elapsed
                ))
    def plot_losses(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.valid_losses, label='Validation loss')
        plt.legend()
        plt.savefig('docs/losses.pdf', format='pdf')
        plt.show()
    
    def save(self):
        torch.save(self.model.state_dict(), 'vae.pt')


