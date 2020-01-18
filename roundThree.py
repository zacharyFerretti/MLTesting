import numpy as nump

#Model that will generate likelihood of complimentary -"ness"
#	of entire palette.
class complimentaryPallete:

	# Shape Of X: 		 [[r0,g0,b0], 
	#					  [r1, g1, b1],
	#					  ...,
	#					  [r7, g7, b7]]
	# Dimensionality of X: [3*8]

	def __init__(self, x, y):
		self.input = x 
		self.weights1 = nump.random.rand(self.input.shape[1],5)
		self.weights2 = nump.random.rand(5,3)
		self.y = y
		self.output = nump.zeros(y.shape)

	def feed_forward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.output = sigmoid(np.dot(self.layer1, self.weights2))

class complimentaryPair:
	# Shape Of X: 		 [[r0,g0,b0], 
	#					  [r1, g1, b1]]
	# Dimensionality of X: [3*2]

	def __init__(self, x, y):
		self.input = x
		self.weights1 = nump.random.rand(self.input.shape[1],5)
		self.weights2 = nump.random.rand(5,3)
		self.y = y
		self.output = nump.zeros(y.shape)

	def feed_forward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.output = sigmoid(np.dot(self.layer1, self.weights2))

x=[[244, 235, 231],[13, 13, 10]]
y=[1]
theModel = complimentaryPair(x,y)
print(theModel.weights1)