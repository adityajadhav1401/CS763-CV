import torch

class Criterion():
	def __init__(self):
		pass

	def forward(self, input, target):
		onehot = torch.DoubleTensor(input.size())
		onehot.zero_()
		onehot.scatter_(1, target.type(torch.long).view(-1,1), 1)
		loss = -1 * (input.softmax(1).clamp(min=1e-150).log() * onehot).sum(1).mean(0)
		return loss

	def backward(self, input, target):
		onehot = torch.DoubleTensor(input.size())
		onehot.zero_()
		onehot.scatter_(1, target.type(torch.long).view(-1,1), 1)
		grad = input.softmax(1) - onehot
		return grad