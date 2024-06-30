from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeModel:
    """
    DecisionTreeModel is a class that represents a decision tree model for classification.

    Args:
        random_state (int): The random seed for reproducibility. Default is 42.
        max_depth (int): The maximum depth of the decision tree. Default is 14.
    """

    def __init__(self, random_state=42, max_depth=14):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    def train(self, X_train, y_train):
        """
        Trains the decision tree model.

        Args:
            X_train (array-like): The input training data.
            y_train (array-like): The target training data.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Predicts the class labels for the given test data.

        Args:
            X_test (array-like): The input test data.

        Returns:
            array-like: The predicted class labels.
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """
        Predicts the class probabilities for the given test data.

        Args:
            X_test (array-like): The input test data.

        Returns:
            array-like: The predicted class probabilities.
        """
        return self.model.predict_proba(X_test)
    
    def importance_plot(self, savefig: str=None):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names=[r'$pT$', r'$\eta$', r'$\phi$', r'$d_0$', r'$d_z$']
        plt.figure(figsize=(10, 6))
        plt.bar(range(5), importances[indices]*100, align='center')
        plt.xticks(range(5), [feature_names[i] for i in indices], rotation=0)
        plt.xlim([-1, 5])
        plt.ylabel('Importance (%)')
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
    

class MLP(nn.Module):
    """
    MLP is a class that represents a multi-layer perceptron neural network.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
    """

    def __init__(self, input_dim, hidden_dims):
        super(MLP, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU(True),
                )
            )
            input_dim = h_dim
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid(),
            )
        )
        self.network = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Performs forward pass through the neural network.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The output tensor.
        """
        return self.network(x)
    


class AE_encoder(nn.Module):
    """
    AE_encoder is a class that represents the encoder part of an autoencoder.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(AE_encoder, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU(True)
                )
            )
            input_dim = h_dim  # Update the input dimension for the next layer
        # Output layer for latent space representation
        modules.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Performs forward pass through the encoder.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The encoded tensor.
        """
        return self.encoder(x)


class AE_decoder(nn.Module):
    """
    AE_decoder is a class that represents the decoder part of an autoencoder.

    Args:
        latent_dim (int): The dimensionality of the latent space representation.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(AE_decoder, self).__init__()
        hidden_dims.reverse()  # Reverse the hidden dimensions for the decoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim, h_dim),
                    nn.ReLU(True)
                )
            )
            latent_dim = h_dim  # Update the input dimension for the next layer
        # Output layer with tanh activation
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        modules.append(nn.Tanh())
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        """
        Performs forward pass through the decoder.

        Args:
            z (tensor): The input tensor.

        Returns:
            tensor: The decoded tensor.
        """
        return self.decoder(z)


class AE(nn.Module):
    """
    AE is a class that represents an autoencoder.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, output_dim):
        super(AE, self).__init__()
        self.encoder = AE_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = AE_encoder(latent_dim, hidden_dims, output_dim)
    
    def forward(self, x):
        """
        Performs forward pass through the autoencoder.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The reconstructed tensor.
        """
        z = self.encoder(x)
        return self.decoder(z)


class VAE_encoder(nn.Module):
    """
    VAE_encoder is a class that represents the encoder part of a variational autoencoder.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE_encoder, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU(True)
                )
            )
            input_dim = h_dim  # Update the input dimension for the next layer
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)  # Output layer for mean
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)  # Output layer for log variance
    
    def forward(self, x):
        """
        Performs forward pass through the encoder.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The mean and log variance tensors.
        """
        hidden = self.encoder(x)
        return self.mu(hidden), self.log_var(hidden)


class VAE_decoder(nn.Module):
    """
    VAE_decoder is a class that represents the decoder part of a variational autoencoder.

    Args:
        latent_dim (int): The dimensionality of the latent space representation.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(VAE_decoder, self).__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim, h_dim),
                    nn.ReLU()
                )
            )
            latent_dim = h_dim  # Update the input dimension for the next layer
        # Add the output layer with tanh activation
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        modules.append(nn.Tanh())
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        """
        Performs forward pass through the decoder.

        Args:
            z (tensor): The input tensor.

        Returns:
            tensor: The decoded tensor.
        """
        return self.decoder(z)
    
    
class VAE(nn.Module):
    """
    VAE is a class that represents a variational autoencoder.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dims (list): A list of integers representing the number of units in each hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAE_decoder(latent_dim, hidden_dims, output_dim)

    def reparameterize(self, mu, log_var):
        """
        Reparameterizes the latent space representation.

        Args:
            mu (tensor): The mean tensor.
            log_var (tensor): The log variance tensor.

        Returns:
            tensor: The reparameterized tensor.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        """
        Performs forward pass through the variational autoencoder.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The reconstructed tensor, mean tensor, and log variance tensor.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var