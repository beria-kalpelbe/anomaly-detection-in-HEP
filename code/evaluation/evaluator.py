import torch
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

with open('code/utils/hyperparameters.json') as f:
    hyperparameters = json.load(f)

BATCH_SIZE = hyperparameters['BATCH_SIZE']

class evaluator(object):
    """
    Evaluate a model on a test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_data (torch.tensor): The test data.
    """
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.scores = None
        
    def run(self):
        """
        Evaluate the model on the test data.
        """
        scores = []
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.test_data):
                x_recon, mu, logvar = self.model(data)
                score = torch.sum(torch.pow(x_recon - data, 2)) / BATCH_SIZE
                scores.append(score)
        scores = torch.tensor(scores)
        self.scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    def get_scores(self):
        """
        Return the scores of the test data.
        """
        return self.scores
    
    def describe_scores(self):
        """
        Return the mean and standard deviation of the scores.
        """
        mean = torch.mean(self.scores)
        std = torch.std(self.scores)
        plt.hist(self.scores.numpy(), bins=150, log=False, density=True, alpha=0.75, label='Histogram')
        density = gaussian_kde(self.scores)
        x_vals = np.linspace(min(self.scores), max(self.scores), 1000)
        plt.plot(x_vals, density(x_vals), linewidth=2, label='Density Curve')
        plt.title(f"Distribution of scores: $\mu=${mean:.2f}, $\sigma=${std:.2f}")
        plt.xlabel('Anomaly score')
        plt.legend()
        plt.savefig('docs/scores-distributions.pdf', format='pdf')
        plt.show()
        
    def roc_curve(self):
        """
        Plot the ROC curve.
        """
        
        pass
        
                