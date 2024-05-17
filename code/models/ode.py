import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
from torch.distributions import Normal
import torch

import pickle

import sys
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


_ = torch.manual_seed(0)



class MultivariateGaussianMDN(nn.Module):
    """
    Multivariate Gaussian MDN with diagonal Covariance matrix.

    For a documented version of this code, see:
    https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py
    """

    def __init__(
        self,
        features,
        hidden_net,
        num_components,
        hidden_features,
    ):
        super().__init__()

        self._features = features
        self._num_components = num_components

        self._hidden_net = hidden_net
        self._logits_layer = nn.Linear(hidden_features, num_components)
        self._means_layer = nn.Linear(hidden_features, num_components * features)
        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )

    def get_mixture_components(self, context):
        h = self._hidden_net(context)

        # mixture coefficients in log space
        logits = self._logits_layer(h)
        logits = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)  # normalization

        # means
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        # log variances for diagonal Cov matrix
        # otherwise: Cholesky decomposition s.t. Cov = AA^T, A is lower triangular.
        log_variances = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )
        variances = torch.exp(log_variances)

        return logits, means, variances
    

import math
def mog_log_prob(theta, logits, means, variances):
    """ Log probability of a mixture of Gaussians.
        args:
            theta: parameters
            logits: log mixture coefficients
            means: means of the Gaussians
            variances: variances of the Gaussians
        returns:
            log probability of the mixture of Gaussians"""
    _, _, theta_dim = means.size()
    theta = theta.view(-1, 1, theta_dim)

    log_cov_det = -0.5 * torch.log(torch.prod(variances, dim=2))

    a = logits
    b = -(theta_dim / 2.0) * math.log(2 * math.pi)
    c = log_cov_det
    d1 = theta.expand_as(means) - means
    precisions = 1.0 / variances
    exponent = torch.sum(d1 * precisions * d1, dim=2)
    exponent = torch.tensor(-0.5) * exponent

    return torch.logsumexp(a + b + c + exponent, dim=-1)


def mog_sample(logits, means, variances):
    """Sample from a mixture of Gaussians.
        args:
            logits: log mixture coefficients
            means: means of the Gaussians
            variances: variances of the Gaussians   
        returns:
            samples from the mixture of Gaussians"""
    
    coefficients = F.softmax(logits, dim=-1)
    choices = torch.multinomial(coefficients, num_samples=1, replacement=True).view(-1)
    chosen_means = means[0, choices, :]  # 0 for first batch position
    chosen_variances = variances[0, choices, :]

    _, _, output_dim = means.shape
    standard_normal_samples = torch.randn(output_dim)
    zero_mean_samples = standard_normal_samples * torch.sqrt(chosen_variances)
    samples = chosen_means + zero_mean_samples

    return samples


class ode():
    def __init__(self, sg_data, bkg_data):
        self.sg_data = data.DataLoader(data.TensorDataset(sg_data, sg_data), batch_size=50)
        self.bkg_data = data.DataLoader(data.TensorDataset(bkg_data, bkg_data), batch_size=50)
        self.num_samples = 1000
        self.num_components = 10
        
    def sg_mdn(self):
        hidden_net = nn.Sequential(
            nn.Linear(5, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
        )

        self.density_signal = MultivariateGaussianMDN(
            features=5,
            hidden_net=hidden_net,
            num_components=5,
            hidden_features=30,
        )
        opt = optim.Adam(self.density_signal.parameters(), lr=0.0001)
        for e in range(5):
            for x_batch, theta_batch in self.sg_data:
                opt.zero_grad()
                logits, means, variances = self.density_signal.get_mixture_components(x_batch)
                log_probs = mog_log_prob(theta_batch, logits, means, variances)
                loss = -log_probs.sum()
                loss.backward()
                opt.step()
    
    def bkg_mdn(self):
        hidden_net = nn.Sequential(
            nn.Linear(5, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
        )

        self.density_bkg = MultivariateGaussianMDN(
            features=5,
            hidden_net=hidden_net,
            num_components=5,
            hidden_features=30,
        )
        opt = optim.Adam(self.density_bkg.parameters(), lr=0.0001)
        for e in range(5):
            for x_batch, theta_batch in self.bkg_data:
                opt.zero_grad()
                logits, means, variances = self.density_bkg.get_mixture_components(x_batch)
                log_probs = mog_log_prob(theta_batch, logits, means, variances)
                loss = -log_probs.sum()
                loss.backward()
                opt.step()
    
    def run(self):
        print("Training signal MDN...")
        self.sg_mdn()
        print("Training background MDN...")
        self.bkg_mdn()
                
    def get_scores(self, data, epsilon=2e-1):
        logits_sg, means_sg, variances_sg = self.density_signal.get_mixture_components(data)
        p_sg = mog_log_prob(data, logits_sg, means_sg, variances_sg)
        
        logits_bkg, means_bkg, variances_bkg = self.density_bkg.get_mixture_components(data)
        p_bkg = mog_log_prob(data, logits_bkg, means_bkg, variances_bkg)
        
        return p_bkg.exp()/(epsilon*p_sg.exp() + (1-epsilon)*p_bkg.exp())
        
    def roc_curve(self, data):
        scores = self.get_scores(data.data)
        fpr, tpr, thresholds = roc_curve(data.labels, scores)
        roc_auc = roc_auc_score(data.labels, scores)
        plt.plot(fpr, tpr)
        plt.title('ROC curve (AUC = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('docs/roc-curve-ode.pdf', format='pdf')
        plt.show()

