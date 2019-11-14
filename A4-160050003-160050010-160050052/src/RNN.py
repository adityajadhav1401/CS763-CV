import torch
import math

torch.manual_seed(0)
dtype = torch.double

class RNN(object):
	def __init__(self, input_dim, hidden_dim, bptt_trunc=20):
		## initialize parameters
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.bptt_trunc = bptt_trunc

		# random initiate the parameters (good)
		self.W_xh = (-2 * math.sqrt(1. / input_dim)) * torch.rand(input_dim, hidden_dim,dtype=dtype) + math.sqrt(1. / input_dim)
		self.W_hh = (-2 * math.sqrt(1. / hidden_dim)) * torch.rand(hidden_dim, hidden_dim,dtype=dtype) + math.sqrt(1. / hidden_dim)
		self.B_h  = (-2 * math.sqrt(1. / hidden_dim)) * torch.rand(hidden_dim,dtype=dtype) + math.sqrt(1. / hidden_dim) 

		# random initiate the parameters (bad)
		# self.W_xh = torch.randn(input_dim, hidden_dim,dtype=dtype)
		# self.W_hh = torch.randn(hidden_dim, hidden_dim,dtype=dtype) 
		# self.B_h  = torch.randn(hidden_dim, dtype=dtype)

		self.H 		   = 0
		self.dLdB_h    = torch.zeros(hidden_dim)
		self.dLdW_xh   = torch.zeros(input_dim,hidden_dim)
		self.dLdW_hh   = torch.zeros(hidden_dim,hidden_dim)


	def forward(self,X):
		[T, D] = X.shape
		self.H = torch.zeros(T+1, self.hidden_dim, dtype=dtype)
		self.H[0,:] = torch.randn(self.hidden_dim)
		for t in range(0,T,1):
			temp1 			= X[t,:].view(1,-1)
			temp2 			= self.H[t,:].view(1,-1)
			self.H[t+1,:] 	= torch.tanh(temp1.mm(self.W_xh.type(torch.float)) + temp2.mm(self.W_hh).type(torch.float) + self.B_h.type(torch.float))

		self.output = self.H[T,:].view(1,-1)
		return self.H[T,:]

	def backward(self,X,dLdO):
		[T, D] = X.shape

		dLdH 		= torch.zeros(T+1, self.hidden_dim)
		dLdH[T,:]	= dLdO
		dLdX 		= torch.zeros(T, self.input_dim)
		
		dLdW_xh = torch.zeros(self.W_xh.shape)
		dLdW_hh = torch.zeros(self.W_hh.shape)
		dLdB_h  = torch.zeros(self.B_h.shape)
			
		for t in range(T-1,-1,-1):
			temp = ((1 - self.H[t+1,:]**2) * dLdH[t+1,:].type(torch.double)).view(1,-1)

			if (T - t <= self.bptt_trunc):
				temp1 			= X[t,:].view(-1,1).type(torch.float)
				temp2 			= self.H[t,:].view(-1,1).type(torch.float)
				dLdB_h  		+= torch.sum(temp,dim=0).type(torch.float)
				dLdW_xh 		+= temp1.mm(temp.view(1,self.hidden_dim).type(torch.float))
				dLdW_hh 		+= temp2.mm(temp.view(1,self.hidden_dim).type(torch.float))

			dLdH[t,:] = temp.mm(self.W_hh.t())
			dLdX[t,:] = temp.mm(self.W_xh.t())

		self.dLdW_xh += dLdW_xh / self.bptt_trunc
		self.dLdW_hh += dLdW_hh / self.bptt_trunc
		self.dLdB_h  += dLdB_h  / self.bptt_trunc

		return dLdX
	
	def update_param(self, lr):
		self.W_xh = self.W_xh - lr * self.dLdW_xh.type(torch.double)
		self.W_hh = self.W_hh - lr * self.dLdW_hh.type(torch.double)
		self.B_h  = self.B_h  - lr * self.dLdB_h.type(torch.double)
		return

	def clear_grad(self):
		self.dLdW_xh = 0
		self.dLdW_hh = 0
		self.dLdB_h  = 0
		return

	def print_param(self):
		print("Weight_xh :")
		print(self.W_xh)
		print("Weight_hh :")
		print(self.W_hh)
		print("Bias_h :")
		print(self.B_h)
		return
	