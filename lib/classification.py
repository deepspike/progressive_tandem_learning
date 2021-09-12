import torch

def training(model, trainloader, optimizer, criterion, device):
	"""Training codes for image classification task"""
	model.train() # Put the model in train mode 

	running_loss = 0.0
	total = 0
	correct = 0 
	for i_batch, (inputs, labels) in enumerate(trainloader, 1):
		# Transfer to GPU
		inputs, labels = inputs.type(torch.FloatTensor).to(device), \
							labels.type(torch.LongTensor).to(device)

		# Model computation and weight update
		y_pred = model.forward(inputs)
		loss = criterion(y_pred, labels)
		_, predicted  = torch.max(y_pred.data, dim=1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
					
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		
	epoch_loss = running_loss/i_batch
	acc_train = correct/total

	return model, acc_train, epoch_loss


def testing(model, testLoader, criterion, device,):
	"""Testing codes for image classification task"""
	model.eval() # Put the model in test mode 

	running_loss = 0.0
	correct = 0
	total = 0
	for data in testLoader:
		inputs, labels = data

		# Transfer to GPU
		inputs, labels = inputs.type(torch.FloatTensor).to(device), \
							labels.type(torch.LongTensor).to(device)

		# forward pass
		y_pred = model.forward(inputs)
		loss = criterion(y_pred, labels)
		_, predicted  = torch.max(y_pred.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		running_loss += loss.item()

	# calculate epoch statistics 	
	epoch_loss = running_loss/len(testLoader)
	acc = correct/total

	return acc, epoch_loss