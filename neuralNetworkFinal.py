from torch import nn
from torch import FloatTensor
from torch import optim
from pandas import read_csv
from ast import literal_eval as make_tuple

from torch.autograd import Variable


class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(6,3)
		self.output = nn.Linear(3, 1)
		print("Hello")
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		# Here we will pass the input through each of our operations
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.output(x)


		#Softmax takes as input a vector of K real numbers, and normalizes
		#	it into a probability distribution of K probabilities.
		return x

def dataProcessing(dataFrameList):
	x = []
	y = []
	for row in dataFrameList:
		firstColor = make_tuple(row[0])
		secondColor = make_tuple(row[1])
		label = row[2]
		array = [float(firstColor[0]*(1/256)), float(firstColor[1]*(1/256)), float(firstColor[2]*(1/256)), float(secondColor[0]*(1/256)), float(secondColor[1]*(1/256)), float(secondColor[2]*(1/256))]
		x.append(array)
		y.append([label])
	return x, y

def main():
	model = NeuralNet()
	print(model)

	#Take in and process our data.
	grandTotal = read_csv('./rgb_labeled_explored.csv', header=None, sep="|").values.tolist()
	x, y = dataProcessing(grandTotal)

	x = FloatTensor(x)
	print(x)
	print(type(x))
	y = FloatTensor(y)

	optimizer = optim.SGD(model.parameters(), lr=0.2)
	loss_func = nn.MSELoss()
	inputs = Variable(x)
	outputs = Variable(y)


	for i in range(1000000):
		prediction = model(inputs)
		loss = loss_func(prediction, outputs)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if i % 10 == 0:
		#	print(loss)

	print(prediction.tolist())

if __name__ == '__main__':
	main()
