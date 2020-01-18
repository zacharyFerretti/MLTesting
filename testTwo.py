import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*3, 5)
        self.fc2 = nn.Linear(5, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)
print(list(net.parameters()))

input = torch.FloatTensor([40, 43, 43])
output=net(input)

print(output)