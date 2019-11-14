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

data = torch.tensor(torchfile.load(data_loc)).double()/255.0
data = data - data.mean(0)
model_config_loc = model_name + "/modelConfig.txt"
model_config = open(model_config_loc,'r')
lines = model_config.readlines()

# Loading the Model from "mdoelConfig.txt"
network = Model()
# network.addLayer(layers.FlattenLayer())
for i in range(1,len(lines)-2):
	words = lines[i].split(" ")
	if(words[0].strip() == "relu"):
		network.addLayer(ReLULayer())

	elif(words[0].strip() == "flatten"):
		network.addLayer(FlattenLayer())

	elif(words[0].strip() == "maxpool"):
		network.addLayer(MaxPoolingLayer([int(words[1]),int(words[2]),int(words[3])],int(words[4]),int(words[5])))

	elif(words[0].strip() == "convo"):
		# words = lines[i].split(" ")
		network.addLayer(ConvolutionLayer([int(words[1]),int(words[2]),int(words[3])],[int(words[4]),int(words[5])],int(words[6]),int(words[7])))

	elif(words[0].strip() == "linear"):
		# words = lines[i].split(" ")
		network.addLayer(LinearLayer(int(words[1]), int(words[2])))

	elif(words[0].strip() == "batchnorm"):
		# words = lines[i].split(" ")
		network.addLayer(BatchNormLayer(int(words[1])))

# print(lines[len(lines)-2].strip())
model_weights_loc = lines[len(lines)-2].strip()
model_bias_loc = lines[len(lines)-1].strip()
# model_weights = torch.tensor(torchfile.load(model_weights_loc))
# model_bias = torch.tensor(torchfile.load(model_bias_loc))
model_weights = torch.load(model_weights_loc)
model_bias = torch.load(model_bias_loc)
# print(model_bias)

# Loading the  weights and biases for the model
count = 0
for i in range(len(network.Layers)):
	layer = network.Layers[i]
	if (isinstance(layer,LinearLayer)):
		layer.W = torch.tensor(model_weights[count]).double()
		# print(model_bias[count].size())
		layer.B = torch.tensor(model_bias[count]).double()
		count += 1
	elif (isinstance(layer,ConvolutionLayer)):
		layer.W = torch.tensor(model_weights[count]).double()
		# print(model_bias[count].size())
		layer.B = torch.tensor(model_bias[count]).double()
		count += 1
	elif (isinstance(layer,BatchNormLayer)):
		layer.Gamma = torch.tensor(model_weights[count]).double()
		# print(model_bias[count].size())
		layer.Beta = torch.tensor(model_bias[count]).double()
		count += 1

activations = network.forward(data.view(-1, 1, 108, 108))
print(activations.size())
predictions = activations.max(1)[1]

torch.save(predictions, model_name+'/testPredictions.bin')
file = open(model_name+'/kaggle.csv','w')
file.write('id,label\n\n')
# print(predictions.size())
id = 0
for prediction in predictions:
	file.write(str(id)+','+str(prediction.item())+'\n')
	id+=1


