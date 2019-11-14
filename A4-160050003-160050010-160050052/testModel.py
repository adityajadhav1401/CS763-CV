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

for i in range(len(sys.argv)):
	if (sys.argv[i] == "-modelName"):
		model_name = sys.argv[i+1]
	elif (sys.argv[i] == "-data"): 
		data_loc = sys.argv[i+1]


batch_size 	= 12
criterion 	= Criterion()
dataset 	= Dataset(batch_size)
dataset.read_data(data_loc,'X_test')
test_data_len = len(dataset.X_test)

model 		= torch.load(model_name + '/model.bin')


Y_pred = torch.zeros(test_data_len)
output_txt = "id,label\n\n"
for i in range(test_data_len):
	data = dataset.one_hot_encode(dataset.X_test[i])
	pred = model.forward(data)
	Y_pred[i] = int(pred.view(1,-1).max(dim=1)[1])
	output_txt += str(i) + "," + str(int(pred.view(1,-1).max(dim=1)[1])) + "\n"

f = open("test_pred.txt", "w")
f.write(output_txt)
f.close

f = open("test_pred.bin", 'wb')
torch.save(Y_pred, f)
f.close()

