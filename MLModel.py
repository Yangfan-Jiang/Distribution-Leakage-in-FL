# Several basic machine learning models
import torch
from torch import nn


class LogisticRegression(nn.Module):
    """Logistic regression model"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)


    
class LightMLP(nn.Module):
    def __init__(self, num_feature, output_size):
        super(LightMLP, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)



class LightMLP1(nn.Module):
    def __init__(self, num_feature, output_size):
        super(LightMLP1, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)

    

class LightMLP2(nn.Module):
    """Deep Neural Network model"""
    def __init__(self, num_feature, output_size):
        super(LightMLP2, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)
    
    
class LightMLP3(nn.Module):
    def __init__(self, num_feature, output_size):
        super(LightMLP3, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.Dropout(0.3),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)
    
    
class LightMLP4(nn.Module):
    def __init__(self, num_feature, output_size):
        super(LightMLP4, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.Dropout(0.4),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.4),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)
    
    
class LightMLP5(nn.Module):
    def __init__(self, num_feature, output_size):
        super(LightMLP5, self).__init__()
        self.h1 = 600
        self.h2 = 100
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.h1),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(self.h2, output_size))

    def forward(self, x):
        return self.model(x)
