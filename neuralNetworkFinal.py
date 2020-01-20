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


	#for i in range(500000):
	for i in range(5000):
	#for i in range(5):
		prediction = model(inputs)
		loss = loss_func(prediction, outputs)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if i % 10 == 0:
		#	print(loss)

	prediction_as_list=prediction.tolist()
	range_counts = [0,0,0,0,0,0,0,0,0,0,0,0]
	range_correct = [0,0,0,0,0,0,0,0,0,0,0,0]
	range_incorrect = [0,0,0,0,0,0,0,0,0,0,0,0]
	entries = []
	i=0
	print(len(y))
	print(outputs)
	for entry in prediction_as_list:
		val = round(entry[0],2)
		
		if(val>0.00 and val<0.05):
			range_counts[0]=range_counts[0]+1
			if(y[i] == 0):
				range_correct[0]=range_correct[0]+1
			else:
				range_incorrect[0]=range_incorrect[0]+1

		elif(val>0.05 and val<0.10):
			range_counts[1]=range_counts[1]+1
			if(outputs[i] == 0):
				range_correct[1]=range_correct[1]+1
			else:
				range_incorrect[1]=range_incorrect[1]+1

		elif(val>0.10 and val<0.15):
			range_counts[2]=range_counts[2]+1
			if(outputs[i] == 0):
				range_correct[2]=range_correct[2]+1
			else:
				range_incorrect[2]=range_incorrect[2]+1

		elif(val>0.15 and val<0.20):
			range_counts[3]=range_counts[3]+1
			if(outputs[i] == 0):
				range_correct[3]=range_correct[3]+1
			else:
				range_incorrect[3]=range_incorrect[3]+1

		elif(val>0.20 and val<0.25):
			range_counts[4]=range_counts[4]+1
			if(outputs[i] == 0):
				range_correct[4]=range_correct[4]+1
			else:
				range_incorrect[4]=range_incorrect[4]+1

		elif(val>0.25 and val<0.30):
			range_counts[5]=range_counts[5]+1
			if(outputs[i] == 0):
				range_correct[5]=range_correct[5]+1
			else:
				range_incorrect[5]=range_incorrect[5]+1
		
		elif(val>0.30 and val<0.35):
			range_counts[6]=range_counts[6]+1
			if(outputs[i] == 0):
				range_correct[6]=range_correct[6]+1
			else:
				range_incorrect[6]=range_incorrect[6]+1
		
		elif(val>0.35 and val<0.40):
			range_counts[7]=range_counts[7]+1
			if(outputs[i] == 1):
				range_correct[7]=range_correct[7]+1
			else:
				range_incorrect[7]=range_incorrect[7]+1
		
		elif(val>0.40 and val<0.45):
			range_counts[8]=range_counts[8]+1
			if(outputs[i] == 1):
				range_correct[8]=range_correct[8]+1
			else:
				range_incorrect[8]=range_incorrect[8]+1
		
		elif(val>0.45 and val<0.50):
			range_counts[9]=range_counts[9]+1
			if(outputs[i] == 1):
				range_correct[9]=range_correct[9]+1
			else:
				range_incorrect[9]=range_incorrect[9]+1
		
		elif(val>0.50 and val<0.55):
			range_counts[10]=range_counts[10]+1
			if(outputs[i] == 1):
				range_correct[10]=range_correct[10]+1
			else:
				range_incorrect[10]=range_incorrect[10]+1
		
		elif(val>0.55):
			range_counts[11]=range_counts[11]+1
			if(outputs[i] == 1):
				range_correct[11]=range_correct[11]+1
			else:
				range_incorrect[11]=range_incorrect[11]+1

	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("~ Statistics About Predictions ~")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print(range_correct)
	print(range_incorrect)
	print("0 - 5 Percent: " + str(range_counts[0])+ " | " + str(range_correct[0]) + "/" + str(range_counts[0]))
	print("5 - 10 Percent: " + str(range_counts[1])+ " | " + str(range_correct[1]) + "/" + str(range_counts[1]))
	print("10 - 15 Percent: " + str(range_counts[2])+ " | " + str(range_correct[2]) + "/" + str(range_counts[2]))
	print("15 - 20 Percent: " + str(range_counts[3])+ " | " + str(range_correct[3]) + "/" + str(range_counts[3]))
	print("20 - 25 Percent: " + str(range_counts[4])+ " | " + str(range_correct[4]) + "/" + str(range_counts[4]))
	print("25 - 30 Percent: " + str(range_counts[5])+ " | " + str(range_correct[5]) + "/" + str(range_counts[5]))
	print("30 - 35 Percent: " + str(range_counts[6])+ " | " + str(range_correct[6]) + "/" + str(range_counts[6]))
	print("35 - 40 Percent: " + str(range_counts[7])+ " | " + str(range_correct[7]) + "/" + str(range_counts[7]))
	print("40 - 45 Percent: " + str(range_counts[8])+ " | " + str(range_correct[8]) + "/" + str(range_counts[8]))
	print("45 - 50 Percent: " + str(range_counts[9])+ " | " + str(range_correct[9]) + "/" + str(range_counts[9]))
	print("50 - 55 Percent: " + str(range_counts[10])+ " | " + str(range_correct[10]) + "/" + str(range_counts[10]))
	print("> 55 Percent: " + str(range_counts[11])+ " | " + str(range_correct[11]) + "/" + str(range_counts[11]))

if __name__ == '__main__':
	main()
