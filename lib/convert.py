import torch
import os 

from lib.utils import save_checkpoint


def freezeLayer(model, layer_list, layer_idx):
	''' freeze the hybrid layer '''

	layer = eval('model.'+str(layer_list[layer_idx]))

	for param in layer.parameters():
		param.requires_grad = False
	
	return model


def netUpdateAcc(model, optimizer, lr, acc_val, best_acc, k, t, epoch, epoch_convert, Tpatient, layer_list, Tsim,
				 ckp_dir, device, buildModel, vthr_list, neuronParam, vthr_convert, stride_list):
	""" Adaptive training schedule to control the training by network validation accuracy (fo regression tasks,
		the validation loss will be used)

	Args:
		model (object): model to update
		optimizer (object): optimizer to update
		lr (float): learning rate
		acc_val (float): validation accuracy
		best_acc (float): best validation accuracy
		k (int): index of the ANN layer to be replaced with a SNN layer
		t (int): patient counter to record the number of epochs that validation loss has not improved
		epoch (int): current training epoch
		epoch_convert (list): epoch at which the ANN layers are converted to SNN layers
		Tpatient (int): the patient period in epochs
		layer_list (list): the layers in the model
		Tsim (int): the encoding window length
		ckp_dir (str): directory to store the checkpoint
		device： cpu or cuda
		buildModel (object): constructor for the hybrid model
		vthr_list (list): firing threshold calculated from the ANN layers prior to the conversion
		neuronParam (dict): configuration of the spiking neuron model
		vthr_convert (list): firing threshold recorded during the network conversion
		stride_list (list): stride for the convolution layers
	Returns:
		model (object): updated model
		optimizer (object): updated optimizer
		k (int): updated index of the ANN layer to be replaced with a SNN layer
		t (int): updated patient counter
		best_acc (float): updated best validation accuracy
		epoch_convert (list): updated epoch at which the ANN layers are converted to SNN layers
		vthr_convert (list): firing threshold recorded during the network conversion
	"""

	# loss not improved
	if acc_val <= best_acc:
		t += 1
		if t < Tpatient:
			return model, optimizer, k, t, best_acc, epoch_convert, vthr_convert
		elif t == Tpatient and k == len(layer_list) - 2:
			print(f'-------------------- Freeze Layer {k + 1}')
			epoch_convert.append(epoch)
			checkpoint = torch.load(os.path.join(ckp_dir, "{0}.pt.tar".format("best")))
			model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with the best acc during the stage
			model = freezeLayer(model, layer_list, k)  # freeze the snn layer
			k += 1
			model = model.to(device)
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1,
										 weight_decay=1e-5)
			t = 0
		elif t == Tpatient and k < len(layer_list) - 2:
			print(f'-------------------- Freeze Layer {k + 1} and Replace Layer {k + 2}')
			epoch_convert.append(epoch)
			checkpoint = torch.load(os.path.join(ckp_dir, "{0}.pt.tar".format("best")))
			model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with the best acc during the stage
			model = freezeLayer(model, layer_list, k)  # freeze the snn layer
			k += 1
			vthr_convert.append(vthr_list[k])
			model = buildModel(model, Tsim, layer_list, k, stride_list, vthr_convert, neuronParam, device)  # replace the ann layer to a hybrid layer
			model = freezeLayer(model, layer_list, k)  # freeze the snn layer
			model = model.to(device)
			print(model)
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
			t = 0  # reset the counter
			best_acc = 0  # reset the best_acc
		else:
			raise ValueError(f"Counter value is incorrect: k = {t}")
	# loss improved, reset counter
	else:
		print('Validation Acc improved --> reset counter')
		save_checkpoint(epoch, model, optimizer, ckp_dir)
		t = 0
		best_acc = acc_val

	return model, optimizer, k, t, best_acc, epoch_convert, vthr_convert


def vthrNorm(model, data_loader, device, percent=99):
	"""Perform threshold normalization to better ultilize the encoding time window"""

	model.eval()

	(inputs, labels) = next(iter(data_loader))  # use one batch of data to estimate the firing threshold

	# Transfer to GPU
	inputs = inputs.type(torch.FloatTensor).to(device)

	# forward pass to get layerwise activation values determined by the percentile
	net_act_max = model.forward(inputs, isCalVnorm=True, percent=percent)

	return net_act_max