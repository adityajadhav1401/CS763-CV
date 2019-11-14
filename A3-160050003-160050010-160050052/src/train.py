import torch
import torchfile
# import cv2
from src.model import *
from src.layers import *
from src.criterion import *

def train(data,labels,network,model_W_loc,model_B_loc):
	dtype = torch.double
	device = torch.device("cpu")

	batch_size, lr, iterations, num_classes = 10, 0.001, 70, 6
	loss = Criterion()

	# data = torch.tensor(torchfile.load('data/data.bin')).double()/255.0
	# labels = torch.tensor(torchfile.load('data/labels.bin'))
	data = data/255.0 # normalize
	# test_data = load_lua('cs763-assign3/test.bin').float()/255.0
	# test_data = test_data-test_data.mean(0)
	# train_data = data[0:2000,:,:]
	# train_labels = labels[0:2000]
	val_data = data[20000:25000,:,:]
	val_data =  (val_data - val_data.mean(0))#/val_data.std(dim=0)*0.1
	val_labels = labels[20000:25000]
	# print(train_data.size())
	# print(train_labels.size())
	train_data = data
	train_data = (train_data - train_data.mean(0))#/train_data.std(dim=0)*0.1
	train_labels = labels
	# print(val_data[0,:,:])
	# print(val_data[3,:,:])
	for iter in range(iterations):
		print(iter)
		train_loss = 0
		for batch in range(train_data.size()[0]//batch_size):
			# network.dispGradParam()
			# activations = network.forward_train(train_data[batch*batch_size:(batch+1)*batch_size,:,:].view(-1, 1, 108, 108))
			activations = network.forward_train(train_data[batch*batch_size:(batch+1)*batch_size,:,:].view(-1, 1, 108, 108))
			train_loss1 = loss.forward(activations, train_labels[batch*batch_size:(batch+1)*batch_size].long())
			network.backward(train_data[batch*batch_size:(batch+1)*batch_size,:,:].view(-1, 1, 108, 108), loss.backward(activations,train_labels[batch*batch_size:(batch+1)*batch_size].long()))
			network.updateParam(lr)
			network.clearGradParam()
			# print(batch)
			# if batch%10==9:
			# 	activations = network.forward(val_data.float().view(-1, 1, 108, 108))
			# 	x = (activations.max(1)[1]-val_labels.long())==0
			# 	print((x.sum().float())*(100.0/val_labels.size()[0]))
			train_loss += train_loss1

		# if iter == 20 or iter == 35 or iter == 55 or iter == 70:
		if iter%11==10:
			lr = 0.1*lr
		# print("-----------")
		activations = network.forward(train_data.view(-1, 1, 108, 108))
		x = (activations.max(1)[1]-train_labels.long())==0
		print((x.sum().float())*(100.0/train_labels.size()[0]))
		# print("-----------")
		activations = network.forward(val_data.view(-1,1, 108, 108))
		x = (activations.max(1)[1]-val_labels.long())==0
		print((x.sum().float())*(100.0/val_labels.size()[0]))

		
		# print(activations)
		print(train_loss)

		# Creating model_W and model_B
		model_W = []
		model_B = []

		for i in range(len(network.Layers)):
			layer = network.Layers[i]
			if(isinstance(layer,LinearLayer)):
				model_W.append(layer.W.tolist())
				model_B.append(layer.B.tolist())
				# print(layer.B)
			if(isinstance(layer,ConvolutionLayer)):
				model_W.append(layer.W.tolist())
				model_B.append(layer.B.tolist())
				# print(layer.B)
			if(isinstance(layer,BatchNormLayer)):
				model_W.append(layer.Gamma.tolist())
				model_B.append(layer.Beta.tolist())
				# print(layer.B)
		torch.save(model_W,model_W_loc)
		torch.save(model_B,model_B_loc)

		# if iter%9==8:
		# test_data = torch.tensor(torchfile.load('data/test.bin')).double()/255.0
		# test_data = test_data-test_data.mean(0)
		# activations = network.forward(test_data.view(-1, 1, 108, 108))
		# activations = activations.max(1)[1]
		# file = open('a.csv','w')
		# file.write('id,label\n\n')
		# id = 0
		# for activation in activations:
		# 	file.write(str(id)+','+str(activation.item())+'\n')
		# 	id+=1
