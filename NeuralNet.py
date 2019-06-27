import numpy as np

#---------------- MISC
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum()

#---------------- Neural Network Class

class NeuralNet:
	def __init__(self, inputSize = 6, weights1 = np.array([]), weights2 = np.array([]), \
				weights3 = np.array([]), NpL1 = 9, NpL2 = 9, outN = 3):
		#NpL: Neurons per layer
		#outN: Number of classes for the output
		self.inputSize = inputSize
		self.weights1 = weights1 if len(weights1) else np.random.rand(self.inputSize, NpL1) #pre-layer1
		self.weights2 = weights2 if len(weights2) else np.random.rand(NpL1,NpL2)#post-layer1
		self.weights3 = weights3 if len(weights3) else np.random.rand(NpL2,outN)#post-layer2
		self.NpL1 = NpL1
		self.NpL2 = NpL2
		self.outN = outN
    
	def feedforward(self, x):
		#print(self.inputSize, self.weights1, self.weights2, self.NpL, self.outN)
		layer1 = sigmoid(np.dot(x, self.weights1))
		layer2 = sigmoid(np.dot(layer1, self.weights2))
		output = sigmoid(np.dot(layer2, self.weights3))
		y = softmax(output)
		index = np.where(y == np.max(y))[0][0] #(array([1]),)
		if index == 0:
			return 'STRAIGHT'
		elif index == 1:
			return 'LEFT'
		elif index == 2:
			return 'RIGHT'
	
	def copy(self):
		return NeuralNet(self.inputSize, self.weights1.copy(), self.weights2.copy(), self.weights3.copy(), self.NpL1, self.NpL2, self.outN)



if __name__ == "__main__": #Testing
	NN1 = NeuralNet()
	NN1.feedforward([1 for k in range(NN1.inputSize)])

	NN2 = NN1.copy()
	NN2.weights2[0][0] += 1
	assert(NN1.weights2[0][0] != NN2.weights2[0][0])
