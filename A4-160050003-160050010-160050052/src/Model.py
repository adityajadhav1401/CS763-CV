import torch
from RNN import *
from Linear import *


class Model(object):
	def __init__(self, nLayers, H, V, D, isTrain):
		self.nLayers = nLayers
		self.H = H
		self.V = V
		self.D = D
		self.isTrain = isTrain
		self.Layers = []
	
	def add_layer(self, layer):
		self.Layers.append(layer)

	def forward(self, input):
		activation = input
		for l in self.Layers:
			activation = l.forward(activation)
		return activation

	def backward(self,input,gradOutput):
		delta = gradOutput.view(1,-1)
		for i in range(len(self.Layers)-1, -1, -1):
			if (i == 0): delta = self.Layers[i].backward(input, delta)
			else: delta = self.Layers[i].backward(self.Layers[i-1].output, delta)
		return

	def clear_grad(self):
		for i in range(len(self.Layers)-1, -1, -1):
			self.Layers[i].clear_grad()		

	def print_param(self):
		for i in range(len(self.Layers)-1, -1, -1):
			self.Layers[i].print_param()

	def update_param(self, lr):
		for l in self.Layers:
			l.update_param(lr)