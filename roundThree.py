import numpy as nump
import pandas

# Model that will generate likelihood of complimentary -"ness"
#	of entire palette.

'''
class complimentaryPallete:

	# Shape Of X: 		 [[r0,g0,b0], 
	#					  [r1, g1, b1],
	#					  ...,
	#					  [r7, g7, b7]]
	# Dimensionality of X: [3*8]

	def __init__(self, x, y):
		self.input = x
		self.weights1 = nump.random.rand(self.input.shape[1], 5)
		self.weights2 = nump.random.rand(5, 3)
		self.y = y
		self.output = nump.zeros(y.shape)

	def feed_forward(self):
		self.layer1 = sigmoid(nump.dot(self.input, self.weights1))
		self.output = sigmoid(nump.dot(self.layer1, self.weights2))
'''

class complimentaryPair:
	# Shape Of X: 		 [[r0,g0,b0], 
	#					  [r1, g1, b1]]
	# Dimensionality of X: [3*2]

	# Activation function
	def sigmoid(t):
		return 1 / (1 + nump.exp(-t))

	# Derivative of sigmoid
	def sigmoid_derivative(p):
		return p * (1 - p)

	def __init__(self, x, y):
		self.input = x
		self.weights1 = nump.random.rand(self.input.shape[1], 5)
		self.weights2 = nump.random.rand(5, 1)
		print(self.weights1)
		print(self.weights2)
		self.y = y
		self.output = nump.zeros(y.shape)

	def feed_forward(self):
		self.layer1 = self.sigmoid(nump.dot(self.input, self.weights1))
		self.output = self.sigmoid(nump.dot(self.layer1, self.weights2))

	def backprop(self):
		d_weights2 = nump.dot(self.layer1.T, (2 * (self.y - self.output) * self.sigmoid_derivative(self.output)))
		d_weights1 = nump.dot(self.input.T, (nump.dot(2 * (self.y - self.output) * self.sigmoid_derivative(self.output),
												  self.weights2.T) * self.sigmoid_derivative(self.layer1)))

		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2

# x=[[244, 235, 231],[13, 13, 10]]
# y = [1]
x = pandas.read_csv('./rgb_two_dominant_with_label.csv', header=None, sep="|")
y = x[2]
print(x)
print("\n~~~\n")
print(y)

theModel = complimentaryPair(x, y)
print("WeightsOne:" + str(theModel.weights1))
print("WeightsTwo:" + str(theModel.weights2))