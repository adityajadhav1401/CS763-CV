import sys
import torchfile
import torch
from src.criterion import Criterion

for i in range(len(sys.argv)):
	if (sys.argv[i] == "-i"):
		input_loc = sys.argv[i+1] 
		# print("/path/to/input.bin") 
	elif (sys.argv[i] == "-t"): 
		target_loc = sys.argv[i+1]
		# print("/path/to/target.bin") 
	elif (sys.argv[i] == "-ig"):
		grad_input_loc = sys.argv[i+1]
 		# print("/path/to/gradInput.bin") 

input_data = torch.tensor(torchfile.load(input_loc)).double()
target_data = torch.tensor(torchfile.load(target_loc) - 1).long() # used 1 indexing in target_data 
criterion = Criterion()
avg_loss = criterion.forward(input_data,target_data)
grad_input_data = criterion.backward(input_data,target_data)

print("Average Loss : " + str(avg_loss.item()))
torch.save(grad_input_data,grad_input_loc)

