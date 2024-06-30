import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import seaborn as sns


def plot_learning_curve(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, color="r", label="Training")
    plt.plot(train_sizes, test_scores_mean, color="g", label="Validation")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.xlabel("Training events")
    plt.ylabel("AUC")
    plt.legend(loc="best")
    
def plot_roc_curve(fpr, tpr, roc_auc, model_name, savefig:bool=False):

    x = np.linspace(0,1,100)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc*100:.2f}%)', color='blue')
    plt.plot(x, x, linestyle='--', label='Random classification', color='black')
    print('ROC curve (AUC = {:.3f}%)'.format(roc_auc*100))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if savefig:
        plt.savefig('plots/{model_name}/{model_name}-roc-curve.pdf', format='pdf')
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, model_name, savefig:bool=False):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Background", "Signal"], yticklabels=["Background", "Signal"])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    if savefig:
        plt.savefig(f'plots/{model_name}/{model_name}-confusion-matrix.pdf', format='pdf')
    plt.show()


def plot_sic(y_true, y_scores, label='SIC curve'):
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    if tps.size == 0 or tps[-1] == 0:
        tpr = np.zeros_like(fps)
        fpr = np.zeros_like(fps)
    else:
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        
    plt.plot(tpr, tpr/np.sqrt(fpr), lw=2, label=label)
    plt.xlabel(r"$\varepsilon_S$")
    plt.ylabel(r"$\varepsilon_S/\sqrt{\varepsilon_B}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    true_positive = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_positive = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    true_negative = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (true_positive + false_negative)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, fpr, fnr, f1_score