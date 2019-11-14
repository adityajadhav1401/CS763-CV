import torch
import math

dtype = torch.double
device = torch.device("cpu")

class LinearLayer:
	def __init__(self, in_neurons, out_neurons):	
		self.output = None
		self.W = torch.randn(out_neurons, in_neurons, device=device, dtype=dtype) / math.sqrt(in_neurons/2)
		self.B = torch.randn(out_neurons, 1, device=device, dtype=dtype) / math.sqrt(in_neurons/2)
		self.gradW = None
		self.gradB = None
		self.gradInput = None
		self.momentW = torch.zeros(out_neurons, in_neurons, device=device, dtype=dtype)
		self.momentB = torch.zeros(out_neurons, 1, device=device, dtype=dtype)
		self.exists = False
		self.dropout = True

	def forward_train(self, input):
		# input  = input*
		# print(self.output.type())
		self.output = input.mm(self.W.t()) + self.B.view(1,-1)
		# print(self.output)
		if self.dropout:
			p=0.1
			self.mask = 1.0 /(1.0 - p) * (torch.rand(self.output.size())>p).double()
			self.output *= self.mask
		return self.output

	def forward(self, input):
		# input  = input*
		# print(input.type())
		# print(self.W.type())
		# print(self.B.type())

		self.output = input.mm(self.W.t()) + self.B.view(1,-1)
		self.mask = torch.ones(self.output.size(), device=device, dtype=dtype)
		# print(self.output)
		return self.output
		
	def backward(self, input, gradOutput):
		# print(gradOutput.size())
		if self.dropout:
			gradOutput = gradOutput * self.mask
		self.gradW = gradOutput.t().mm(input)#+ 0.1*self.W
		# print(self.W.norm())
		self.gradB = torch.sum(gradOutput, 0).view(-1,1)# + 0.1*self.B
		self.gradInput = gradOutput.mm(self.W)
		return self.gradInput

	def disp(self):
		print("Weights")
		print(self.W)
		print("Biases")
		print(self.B)

	def clearGrad(self):
		self.gradW = None
		self.gradB = None

	def updateParam(self, lr):
		if not self.exists:
			self.momentW = self.gradW
			self.momentB = self.gradB
			self.exists = True
		else:
			self.momentW = 0.9 * self.momentW + 0.1 * self.gradW
			self.momentB = 0.9 * self.momentB + 0.1 * self.gradB
		self.W -= lr * self.momentW
		self.B -= lr * self.momentB




class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, num_filters, stride):	
		self.output = None
		self.gradW = None
		self.gradB = None
		self.gradInput = None
		
		self.stride = stride
		self.filter_row, self.filter_col = filter_size
		self.num_filters = num_filters
		self.in_depth, self.in_row, self.in_col = in_channels
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		self.W = torch.randn(self.num_filters, self.in_depth, self.filter_row, self.filter_col, device=device, dtype=dtype)
		self.B = torch.randn(self.num_filters)
 
	def forward(self, input):
		self.output = torch.zeros(input.size()[0], self.num_filters, self.out_row, self.out_col)
		for row in range(self.out_row):
			for col in range(self.out_col):
				temp = input[:,:,row*self.stride:row*self.stride+self.filter_row,  col*self.stride:col*self.stride+self.filter_col].view(-1,1, self.in_depth, self.filter_row, self.filter_col).repeat(1,self.num_filters, 1,1,1) * self.W
				self.output[:,:, row,col] = temp.sum((2,3,4))
		self.output += self.B.view(1,self.num_filters, 1, 1).repeat(1,1,self.out_row, self.out_col)
		# print(self.output)
		return self.output
		
	def backward(self, input, gradOutput):
		# print(gradOutput.size())
		self.gradW = torch.zeros(self.W.size())
		self.gradB = torch.zeros(self.B.size())
		self.gradInput = torch.zeros(input.size())
		for row in range(self.filter_row):
			for col in range(self.filter_col):
				temp = input[:, :, row:row+self.out_row*self.stride:self.stride, col:col+self.out_col*self.stride:self.stride].view(-1,1, self.in_depth, self.out_row, self.out_col).repeat(1,self.num_filters,1,1,1) * gradOutput.view(-1,self.num_filters, 1, self.out_row, self.out_col).repeat(1,1,self.in_depth,1,1) 
				self.gradW[:,:,row, col] = temp.sum((0,3,4))

		for row in range(self.out_row):
			for col in range(self.out_col):
				temp = gradOutput[:,:,row,col].view(-1,self.num_filters, 1 ,1 ,1).repeat(1,1,self.in_depth, self.filter_row, self.filter_col)*self.W 
				self.gradInput[:,:, row*self.stride:row*self.stride+self.filter_row, col*self.stride:col*self.stride+self.filter_col] += temp.sum(1)

		self.gradB = gradOutput.sum((0,2,3))
		return self.gradInput

	def disp(self):
		print("Weights")
		print(self.W)
		print("Biases")
		print(self.B)

	def clearGrad(self):
		self.gradW = None
		self.gradB = None

	def updateParam(self, lr):
		self.W -= lr * self.gradW
		self.B -= lr * self.gradB

class MaxPoolingLayer:
	def __init__(self, in_channels, filter_row, filter_col):	
		self.output = None
		self.gradInput = None
		self.filter_row = filter_row
		self.filter_col = filter_col
		self.out_row, self.out_col = in_channels[1]//filter_row, in_channels[2]//filter_col

	def forward(self, input):
		self.output = torch.zeros(input.size()[0], input.size()[1], self.out_row, self.out_col)
		for row in range(self.out_row):
			for col in range(self.out_col):
				# print(input[:,:,row*self.filter_row:(row+1)*self.filter_row, col*self.filter_col:(col+1)*self.filter_col].max(3)[0].max(2)[0].size())
				self.output[:,:,row,col] = input[:,:,row*self.filter_row:(row+1)*self.filter_row, col*self.filter_col:(col+1)*self.filter_col].max(3)[0].max(2)[0]
		# print(self.output)
		return self.output
		
	def backward(self, input, gradOutput):
		# print(gradOutput.size())
		self.gradInput = torch.zeros(input.size())
		for row in range(input.size()[2]):
			for col in range(input.size()[3]):
				self.gradInput[:,:, row, col] = gradOutput[:,:,row//self.filter_row, col//self.filter_col]*(self.output[:,:,row//self.filter_row, col//self.filter_col]==input[:,:,row, col]).float()
		return self.gradInput

	def disp(self):
		return 
	def clearGrad(self):
		return 
	def updateParam(self, lr):
		return 

class BatchNormLayer:

	def __init__(self, in_neurons):	
		self.output = None
		self.Gamma = torch.randn(in_neurons, device=device, dtype=dtype)
		self.Beta = torch.randn(in_neurons, device=device, dtype=dtype)
		self.gradGamma = None
		self.gradBeta = None
		self.gradInput = None
		self.RunningStd = None
		self.RunningBeta = None
		self.momentG = torch.zeros(in_neurons, device=device, dtype=dtype)
		self.momentB = torch.zeros(in_neurons, device=device, dtype=dtype)
		self.exists = False
		self.exists1 = False
		self.momentum = 0.9

	def forward_train(self, input):
		# print(input.std(dim=0))
		temp = (input-input.mean(0))/(input.std(dim=0).clamp(min=1e-150))
		self.output = temp*self.Gamma + self.Beta
		return self.output

	def forward(self, input):
		if not self.exists:
			self.RunningStd = input.std(dim=0)
			self.RunningBeta = input.mean(0)
			self.exists = True
		else:
			self.RunningStd = self.momentum * self.RunningStd + (1.0-self.momentum)*input.std(dim=0)
			self.RunningBeta = self.momentum * self.RunningBeta + (1.0-self.momentum)*input.mean(0)
		temp = (input-self.RunningBeta)/self.RunningStd
		self.output = temp*self.Gamma + self.Beta
		return self.output
		
	def backward(self, input, gradOutput):
		gradNormInput = gradOutput*self.Gamma
		inputMean = (input - input.mean(0))
		invInputStd = 1.0/input.std(dim=0)
		self.gradGamma = (inputMean * invInputStd * gradOutput).sum(0)
		self.gradBeta = gradOutput.sum(0)
		gradStd = (gradNormInput*inputMean).sum(0) * (-0.5) * invInputStd**3 
		gradMean = gradNormInput*(-1.0*invInputStd) + gradStd * (-2 * inputMean).mean(0)
		self.gradInput = (gradNormInput*invInputStd) + (gradStd*2*inputMean/input.size(0)) +  gradMean/input.size(0) 
		return self.gradInput

	def disp(self):
		print("Gamma")
		print(self.Gamma)
		print("Beta")
		print(self.Beta)
		return

	def clearGrad(self):
		self.gradGamma = None
		self.gradBeta = None
		return

	def updateParam(self, lr):
		if not self.exists1:
			self.momentG = self.gradGamma
			self.momentB = self.gradBeta
			self.exists1 = True
		else:
			self.momentG = 0.9 * self.momentG + 0.1 * self.gradGamma
			self.momentB = 0.9 * self.momentB + 0.1 * self.gradBeta
		self.Gamma -= lr * self.momentG
		self.Beta -= lr * self.momentB
		return

class ReLULayer:
	def forward(self, input):
		self.output = input.clamp(min=0)
		return self.output

	def forward_train(self, input):
		self.output = input.clamp(min=0)
		return self.output
		
	def backward(self, input, gradOutput):
		self.gradInput = input.sign().clamp(min=0) * gradOutput
		return self.gradInput

	def disp(self):
		return

	def clearGrad(self):
		return

	def updateParam(self, lr):
		return

class FlattenLayer:
	def forward(self, input):
		self.output = input.view(input.size()[0],-1)
		return self.output

	def forward_train(self, input):
		self.output = input.view(input.size()[0],-1)
		return self.output
		
	def backward(self, input, gradOutput):
		self.gradInput = gradOutput.view(input.size())
		return self.gradInput

	def disp(self):
		return

	def clearGrad(self):
		return

	def updateParam(self, lr):
		return