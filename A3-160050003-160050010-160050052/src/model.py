import torch

dtype = torch.double
device = torch.device("cpu")

from src.layers import *

class Model:

	def __init__(self):
		self.Layers = []
		self.isTrain = None
		# self.alpha = alpha
		# self.batchSize = batchSize
		# self.epochs = epochs
		# self.out_nodes = out_nodes

	def addLayer(self, layer):
		self.Layers.append(layer)

	def forward(self, input):	
		activation = input
		for l in self.Layers:
			activation = l.forward(activation)
		# print(activation)
		return activation

	def forward_train(self, input):	
		activation = input
		for l in self.Layers:
			if l!=self.Layers[-1]:
				activation = l.forward_train(activation)
			else:
				activation = l.forward(activation)
		# print(activation)
		return activation

	def backward(self, input, gradOutput):
		delta = gradOutput
		for i in range(len(self.Layers)-1, -1, -1):
			if i != 0:
				delta = self.Layers[i].backward(self.Layers[i-1].output, delta)
			else:
				delta = self.Layers[i].backward(input, delta)

	def dispGradParam(self):
		for i in range(len(self.Layers)-1, -1, -1):
			self.Layers[i].disp()

	def clearGradParam(self):
		for i in range(len(self.Layers)-1, -1, -1):
			self.Layers[i].clearGrad()

	def updateParam(self, lr):
		for l in self.Layers:
			l.updateParam(lr)
