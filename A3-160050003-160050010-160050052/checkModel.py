import sys
import torchfile
import torch
from src.model import Model
from src.layers import *
from src.train import *

for i in range(len(sys.argv)):
	if (sys.argv[i] == "-config"):
		model_config_loc = sys.argv[i+1]
	 	# print("/path/to/modelConfig.txt") end
	elif (sys.argv[i] == "-i"): 
		input_loc = sys.argv[i+1]
		# print("/path/to/input.bin") end
	elif (sys.argv[i] == "-og"): 
		grad_output_loc = sys.argv[i+1]
		# print("/path/to/gradOutput.bin") end
	elif (sys.argv[i] == "-o"): 
		output_loc = sys.argv[i+1]
		# print("/path/to/output.bin") end
	elif (sys.argv[i] == "-ow"): 
		grad_weight_loc = sys.argv[i+1]
		# print("/path/to/gradWeight.bin") end
	elif (sys.argv[i] == "-ob"): 
		grad_b_loc = sys.argv[i+1]
		# print("/path/to/gradB.bin") end
	elif (sys.argv[i] == "-ig"): 
		grad_input_loc = sys.argv[i+1]
		# print("/path/to/gradInput.bin") end

# Create the model using "modelConfig.txt"
model_config = open(model_config_loc,'r')
lines = model_config.readlines()

network = Model()
network.addLayer(FlattenLayer())
for i in range(1,len(lines)-2):
	if(lines[i].strip() == "relu"):
		network.addLayer(ReLULayer())
	else:
		words = lines[i].split(" ")
		network.addLayer(LinearLayer(int(words[1]), int(words[2])))
model_weights_loc = lines[len(lines)-2].strip()
model_bias_loc = lines[len(lines)-1].strip()
model_weights = torchfile.load(model_weights_loc)
model_bias = torchfile.load(model_bias_loc)
# print(model_bias)
# print(model_weights)

count = 0
for i in range(len(network.Layers)):
	layer = network.Layers[i]
	if (isinstance(layer,LinearLayer)):
		layer.W = torch.tensor(model_weights[count])
		# print(model_bias[count].size())
		layer.B = torch.tensor(model_bias[count])
		count += 1


# Load "input.bin" and "gradOutput.bin"
input = torch.tensor(torchfile.load(input_loc))
grad_output = torch.tensor(torchfile.load(grad_output_loc))

# Setting batch_size to number of data points in sample "input.bin"
batch_size = len(input)
# Performing forward_train and backward_train on the model
activation = network.forward(input.view(-1, input.size()[1], input.size()[2], input.size()[3]))
network.clearGradParam()
network.backward(input.view(-1, input.size()[1],input.size()[2], input.size()[3]), grad_output)

# Creating grad_weight, grad_b,
grad_weight = []
grad_b = []

count = 0
for i in range(len(network.Layers)):
	layer = network.Layers[i]
	if (isinstance(layer,LinearLayer)):
		grad_weight.append(layer.gradW.tolist()) 
		grad_b.append(layer.gradB.view(-1).tolist())
		count += 1

# Creating output
output = network.Layers[len(network.Layers)-1].output

# Creating grad_input
grad_input = network.Layers[0].gradInput

# Saving to bin files
torch.save(output,output_loc)
torch.save(grad_weight,grad_weight_loc)
torch.save(grad_b,grad_b_loc)
torch.save(grad_input,grad_input_loc)



		



