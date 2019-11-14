import sys
import torchfile
import torch
import os
from src.train import *
from src.layers import *
from src.model import Model

for i in range(len(sys.argv)):
	if (sys.argv[i] == "-modelName"):
		model_name = sys.argv[i+1] 
		# print("Folder Name") 
	elif (sys.argv[i] == "-data"): 
		data_loc = sys.argv[i+1]
		# print("/path/to/train/data.bin") 
	elif (sys.argv[i] == "-target"):
		target_loc = sys.argv[i+1]
 		# print("/path/to/target/labels.bin") 

if not os.path.exists(model_name):
    os.makedirs(model_name)

data = torch.tensor(torchfile.load(data_loc)).double()
target = torch.tensor(torchfile.load(target_loc))

# data = data[:250,:,:]
# data = data[:250]

# Location for storing model params
model_config_loc = model_name + "/modelConfig.txt"
model_W_loc = model_name + "/W.bin"
model_B_loc = model_name + "/B.bin"	

# Creating a bestModel
network = Model()
network.addLayer(FlattenLayer())
network.addLayer(LinearLayer(11664, 128))
# network.addLayer(BatchNormLayer(64))
network.addLayer(ReLULayer())
network.addLayer(LinearLayer(128, 6))

# Creating model_config
model_config = ""
count = 0
for i in range(len(network.Layers)): 
	layer = network.Layers[i]
	if(isinstance(layer,LinearLayer) or isinstance(layer,ConvolutionLayer)):
		count += 1

model_config += str(count) + "\n"

for i in range(len(network.Layers)): 
	layer = network.Layers[i]
	if(isinstance(layer,LinearLayer)):
		model_config += "linear " +str(layer.W.size()[1]) + " " + str(layer.W.size()[0]) + "\n"
	if(isinstance(layer,ConvolutionLayer)):
		model_config += "convo " +str(layer.in_depth) + " " + str(layer.in_row) + \
						" " + str(layer.in_col) + " " + str(layer.filter_row) + " " + str(layer.filter_col) + \
						" " + str(layer.num_filters) + " " + str(layer.stride) +  "\n"
	if(isinstance(layer,MaxPoolingLayer)):
		model_config += "maxpool " +str(layer.in_depth) + " " + str(layer.in_row) + \
						" " + str(layer.in_col) + " " + str(layer.filter_row) + " " + str(layer.filter_col) + "\n"
	if(isinstance(layer,ReLULayer)):
		model_config += "relu" + "\n"
	if(isinstance(layer,BatchNormLayer)):
		model_config += "batchnorm " + str(layer.Gamma.size()[0]) +"\n"
	if(isinstance(layer,FlattenLayer)):
		model_config += "flatten" + "\n"

model_config += model_W_loc + "\n"
model_config += model_B_loc + "\n"
file = open(model_config_loc,'w')
file.write(model_config)
file.close()

# Training the bestModel
train(data,target,network,model_W_loc,model_B_loc)

