import torch

device = torch.device("cpu")

class Criterion:
	def forward(self, input, target):
		onehot = torch.DoubleTensor(input.size())
		onehot.zero_()
		onehot.scatter_(1, target.view(-1,1), 1)
		return -1 * (input.softmax(1).clamp(min=1e-150).log() * onehot).sum(1).mean(0)
		
	def backward(self, input, target):
		onehot = torch.DoubleTensor(input.size())
		onehot.zero_()
		onehot.scatter_(1, target.view(-1,1), 1)
		# print(input.softmax - onehot)
		return input.softmax(1) - onehot