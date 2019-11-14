import torch
import math 

dtype = torch.double
device = torch.device("cpu")

class Linear():
	def __init__(self, in_neurons, out_neurons):
		self.output = None
		self.weight = torch.randn(out_neurons, in_neurons, device=device, dtype=dtype) / math.sqrt(in_neurons/2)
		self.bias = torch.randn(out_neurons, 1, device=device, dtype=dtype) / math.sqrt(in_neurons/2)
		self.gradWeight = None
		self.gradBias = None
		self.gradInput = None
		self.momentW = torch.zeros(out_neurons, in_neurons, device=device, dtype=dtype)
		self.momentB = torch.zeros(out_neurons, 1, device=device, dtype=dtype)
		self.exists = False

	def forward(self,input):
		self.output = input.view(1,-1).mm(self.weight.t()) + self.bias.view(1,-1)
		self.mask = torch.ones(self.output.size(), device=device, dtype=dtype)
		return self.output

	def backward(self, input, gradOutput):
		self.gradWeight = gradOutput.t().mm(input)
		self.gradBias = torch.sum(gradOutput, 0).view(-1,1)
		self.gradInput = gradOutput.mm(self.weight)
		return self.gradInput

	def print_param(self):
		print("Weights :")
		print(self.weight)
		print("Bias :")
		print(self.bias)
		return

	def clear_grad(self):
		self.gradWeight = 0
		self.gradBias = 0
		return

	def weights_norm(self):
		return torch.norm(self.weight) + torch.norm(self.bias)

	def update_param(self, lr):
		if not self.exists:
			self.momentW = self.gradWeight
			self.momentB = self.gradBias
			self.exists = True
		else:
			self.momentW = 0.9 * self.momentW + 0.1 * self.gradWeight
			self.momentB = 0.9 * self.momentB + 0.1 * self.gradBias
		self.weight -= lr * self.momentW
		self.bias -= lr * self.momentB
		return