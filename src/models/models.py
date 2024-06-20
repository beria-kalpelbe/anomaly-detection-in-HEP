from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn


class DecisionTreeModel:
    def __init__(self, random_state=42, max_depth=14):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    


class MLP(nn.Module):
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
        return self.network(x)
    
