import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 18)
        self.hidden2 = nn.Linear(18, 9)
        self.hidden3 = nn.Linear(9, 3)
        self.output = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Here we will pass the input through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        x = self.output(x)

        # Softmax takes as input a vector of K real numbers, and normalizes
        #   it into a probability distribution of K probabilities.
        return x
