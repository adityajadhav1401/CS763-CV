import torch

class Dataset:
	def __init__(self,batch_size):
		self.vocab =   ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '15', '19', '20', '21', '26', '27', '30', '33', '37', '38', '39', '40', '41', '42', '43', '45', '54', '57',    	\
						'60', '63', '64', '65', '66', '75', '77', '78', '83', '85', '91', '93', '94', '96', '97', '99', '102', '104', '110', '114', '117', '118', '119', '120', '122', '125', '128', 	\
						'132', '133', '140', '141', '142', '143', '144', '146', '148', '155', '156', '157', '158', '159', '160', '162', '163', '168', '172', '173', '174', '175', '176', '179',     	\
						'180', '183', '184', '185', '191', '192', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '211', '212', 		\
						'213', '214', '219', '220', '221', '224', '226', '228', '229', '230', '231', '233', '234', '240', '242', '243', '252', '254', '255', '256', '258', '259', '260', '264', 		\
						'265', '266', '268', '269', '270', '272', '289', '292', '293', '295', '298', '300', '301', '307', '308', '309', '311', '314', '320', '322', '324', '331', '332', '340']
		self.vocab_size = 153
		self.vocab_enc  = {}
		self.word_dim   = 153
		self.batch_size = 1
		self.num_batch  = 0
		self.X_train   	= []
		self.Y_train 	= []
		self.X_test  	= []
		self.batch_size = batch_size
		self.num_batch 	= (len(self.X_train) // self.batch_size) + 1
		for i, l in enumerate(self.vocab): self.vocab_enc[l] = i

	def one_hot_encode(self,X):
		newX = torch.zeros(len(X),(self.vocab_size))
		for i in range(len(X)): newX[i,self.vocab_enc[str(X[i].item())]] = 1
		return newX

	def read_data(self, location, data_type):
		f1 = open(location, 'r')
		if (data_type == 'X_train'):
			for line in f1.readlines(): self.X_train.append(torch.tensor(list(map(int,line.strip().split(" ")))).type(torch.int))
		elif (data_type == 'Y_train'):
			for line in f1.readlines(): self.Y_train.append(torch.tensor(list(map(int,line.strip().split(" ")))).type(torch.int))
		elif (data_type == 'X_test'):
			for line in f1.readlines(): self.X_test.append(torch.tensor(list(map(int,line.strip().split(" ")))).type(torch.int))

	def get_batch(self, index):
		start_index = index * self.batch_size
		end_index = min((index + 1) * self.batch_size, len(self.X_train))
		
		# return a list of tensors
		return_type = 'simple'
		batch = []
		batch_labels = []

		for i in range(start_index,end_index):
			datum = self.X_train[i]
			datum_one_hot = self.one_hot_encode(datum)
			batch.append(datum_one_hot)
			batch_labels.append(self.Y_train[i])

		# encoding with 0 padded values in beginning
		# return_type = 'complex'
		# max_sentence_len = max(map(lambda x: self.X_train[x].shape[0],range(start_index,end_index)))

		# batch = np.zeros((max_sentence_len, self.vocab_size, self.batch_size))
		# batch_labels = np.zeros(self.batch_size)

		# for i in range(start_index,end_index):
		# 	datum = self.X_train[i]
		# 	datum = np.append([0] * (max_sentence_len - len(datum)), datum, axis=0)
		# 	datum_one_hot = self.one_hot_encode(datum,self.vocab)
		# 	batch[:,:,(i-start_index)] = datum_one_hot.T	
		# 	batch_labels[(i-start_index)] = self.Y_train[(i-start_index)]

		return [batch, batch_labels, return_type] 