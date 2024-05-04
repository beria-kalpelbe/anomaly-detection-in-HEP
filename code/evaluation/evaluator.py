import torch
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import seaborn as sns

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
    def __init__(self, model, test_data, labels):
        self.model = model
        self.test_data = test_data
        self.labels = labels
        self.scores = None
        
    def run(self):
        """
        Evaluate the model on the test data.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)
        self.model.to(device)
        scores = []
        self.model.eval()

        with torch.no_grad():
            for idx, data in enumerate(self.test_data):
                data = data.to(device)
                x_recon, mu, logvar = self.model(data)
                score = torch.sum(torch.pow(x_recon - data, 2)) / BATCH_SIZE
                scores.append(score)
        self.scores = torch.tensor(scores)
        self.scores = (self.scores - self.scores.min()) / (self.scores.max() - self.scores.min())
        self.sg_scores = self.scores[self.labels == 1]
        self.bk_scores = self.scores[self.labels == 0]
    
    def get_scores(self):
        """
        Return the scores of the test data.
        """
        return self.scores, self.sg_scores, self.bk_scores
    
    def describe_scores(self):
        """
        Return the mean and standard deviation of the scores.
        """
        mean = torch.mean(self.sg_scores)
        std = torch.std(self.sg_scores)
        plt.hist(self.sg_scores.numpy(), bins=150, log=False, density=True, alpha=0.75, label='Signal')
        density = gaussian_kde(self.sg_scores)
        x_vals = np.linspace(min(self.sg_scores), max(self.sg_scores), 1000)
        plt.plot(x_vals, density(x_vals), linewidth=2, label='Density signal')
        
        mean = torch.mean(self.bk_scores)
        std = torch.std(self.bk_scores)
        plt.hist(self.bk_scores.numpy(), bins=150, log=False, density=True, alpha=0.75, label='Background')
        density = gaussian_kde(self.bk_scores)
        x_vals = np.linspace(min(self.bk_scores), max(self.bk_scores), 1000)
        plt.plot(x_vals, density(x_vals), linewidth=2, label='Density background')
        
        
        plt.title(f"Distribution of normalized anomaly scores: $\mu=${mean:.2f}, $\sigma=${std:.2f}")
        plt.xlabel('Anomaly score')
        plt.legend()
        plt.savefig('docs/scores-distributions.pdf', format='pdf')
        plt.show()
        
    def roc_curve(self, type_model:str="bdt"):
        """
        Plot the ROC curve.
        """
        if type_model == "bdt":
            probs = self.model.predict_proba(self.test_data)
            fpr, tpr, thresholds = roc_curve(self.labels, probs[:,1])
            roc_auc = roc_auc_score(self.labels, probs[:,1])
            plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.savefig(f'docs/roc-curve-{type_model}.pdf', format='pdf')
            plt.show()
    
    def confusion_matrix(self, type_model:str="bdt"):
        """
        Plot the confusion matrix.
        """
        y_pred = self.model.predict(self.test_data)
        cm = confusion_matrix(self.labels, y_pred)
        accuracy = accuracy_score(self.labels, y_pred)*100
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix (Accuracy: {:.2f}%)".format(accuracy))
        plt.savefig(f'docs/cm-{type_model}.pdf', format='pdf')
        plt.show()
        
                