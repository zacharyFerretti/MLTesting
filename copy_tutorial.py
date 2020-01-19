from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer = torch.nn.Linear(1, 1)

	def forward(self, x):
		x = self.layer(x)
		return x

net = Net()

x = np.random.rand(100)
y = np.sin(x) * np.power(x,3) + 3*x + np.random.rand(100)*0.8

plt.scatter(x, y)
#plt.show()

x = torch.from_numpy(x.reshape(-1,1)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

print(x)
print(type(x))
inputs = Variable(x)
print(inputs)
print(type(inputs))
outputs = Variable(y)

for i in range(250):
	prediction = net(inputs)
	loss = loss_func(prediction, outputs)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if(i%25==0):
		print(loss)