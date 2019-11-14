import sys
import torch
import os
sys.path.append('src')
from Model import Model
from Dataset import Dataset
from Criterion import Criterion
from RNN import RNN
from Linear import Linear

torch.manual_seed(1)

def train(epoches,lr):
	global model,train_data_len, dataset
	
	for iter in range(int(epoches)):
		# batch permutation
		# rand_perm = torch.randperm(dataset.num_batch)
		# for b in range(dataset.num_batch):
		# 	b_permed = rand_perm[b]
		# 	batch_loss = 0
		# 	[batch, batch_labels, return_type] = dataset.get_batch(b_permed) 
		# 	for i in range(dataset.batch_size):
		# 		data = batch[i]
		# 		Y_pred = model.forward(data)
		# 		loss = criterion.forward(Y_pred, torch.DoubleTensor(batch_labels[i]))
		# 		grad_loss = criterion.backward(Y_pred, batch_labels[i].type(torch.double))
		# 		batch_loss += loss
		# 		model.backward(data,grad_loss)
		# 	model.update_param(lr/dataset.batch_size)
		# 	model.clear_grad()
		# 	print("Batch : " + str(b))
		# 	print("Batch Loss : " str(batch_loss/dataset.batch_size))
		# 	batch_loss = 0

		# complete data permutation
		rand_perm = torch.randperm(train_data_len)
		batch_loss = 0
		ctr = 0
		for j in range(train_data_len):
			ctr += 1
			i = rand_perm[j]
			data = dataset.one_hot_encode(dataset.X_train[i])
			Y_pred = model.forward(data)
			loss = criterion.forward(Y_pred, dataset.Y_train[i].type(torch.double))
			grad_loss = criterion.backward(Y_pred, dataset.Y_train[i].type(torch.double))
			batch_loss += loss
			model.backward(data,grad_loss)
			if (ctr == dataset.batch_size):
				model.update_param(lr/dataset.batch_size)
				model.clear_grad()
				print("Batch : " + str((train_data_len*iter+j)//dataset.batch_size))
				print("Batch Loss : " + str((batch_loss/dataset.batch_size).item()))
				batch_loss = 0
				ctr = 0

def accuracy(start,len):
	count = 0
	for i in range(len):
		data = dataset.one_hot_encode(dataset.X_train[start+i])
		Y_pred = model.forward(data)
		count += (int(Y_pred.view(1,-1).max(dim=1)[1]) == int(dataset.Y_train[start+i].item()))
	print(count/len)


# main_program
for i in range(len(sys.argv)):
	if (sys.argv[i] == "-modelName"):
		model_name = sys.argv[i+1] 
	elif (sys.argv[i] == "-data"): 
		data_loc = sys.argv[i+1] 
	elif (sys.argv[i] == "-target"):
		target_loc = sys.argv[i+1] 

if not os.path.exists(model_name):
    os.makedirs(model_name)

batch_size 	= 12
criterion 	= Criterion()
dataset 	= Dataset(batch_size)
model 		= Model(2,128,153,153,True)

dataset.read_data(data_loc,'X_train')
dataset.read_data(target_loc,'Y_train')
train_data_len = len(dataset.X_train)

model.add_layer(RNN(153,128,20))
model.add_layer(Linear(128,2))

train(8,1)
train(3,1e-1)
accuracy(0,train_data_len)
train(6,1e-2)
accuracy(0,train_data_len)
train(3,1e-3)
accuracy(0,train_data_len)
train(8,1e-6)
accuracy(0,train_data_len)
train(3,1e-7)
accuracy(0,train_data_len)

file = open(model_name + '/model.bin', 'wb')
torch.save(model, file)
file.close()

