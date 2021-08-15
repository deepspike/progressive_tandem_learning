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
	""" Adaptive training schedule update

	Args:
		model (object): model to update
		optimizer (object): optimizer to update
		loss (float): validation loss of the current epoch
		loss_old (float): validation loss of the previous epoch
		k (int): index of the ANN layer to be replaced with a SNN layer
		t (int): counter to record the number of epochs that validation loss is not improved
		Tpatient (int): patient epochs
		layer_list (list): the layers in the model
		Tsim (int): encoding window length
		device： cpu or gpu

	Returns:
		model: updated model
		optimizer: updated optimizer
		k: updated ANN layer index to be replaced with a SNN layer
		t: updated counter
		best_loss: updated best_loss
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
			model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with best_loss
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
			model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with best_loss
			model = freezeLayer(model, layer_list, k)  # freeze the snn layer
			k += 1
			vthr_convert.append(vthr_list[k])
			print("Updated neuron threshold of each layer ", vthr_convert)
			model = buildModel(model, Tsim, layer_list, k, stride_list, vthr_convert,
							   neuronParam, device)  # replace the ann layer to hybrid layer
			model = freezeLayer(model, layer_list, k)  # freeze the snn layer (no tandem learning)
			model = model.to(device)
			print(model)
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
										 weight_decay=1e-5)
			t = 0  # reset the counter
			best_acc = 0  # reset the best_loss to a high value
		else:
			raise ValueError(f"Counter value is incorrect: k = {t}")
	# loss improved, reset counter
	else:
		print('Validation Acc improved --> reset counter')
		save_checkpoint(epoch, model, optimizer, ckp_dir)
		t = 0
		best_acc = acc_val

	return model, optimizer, k, t, best_acc, epoch_convert, vthr_convert

def vthrNorm(model, data_loader, T, device, percent=99):
	"""Perform wieght normalization to better ultilize the encoding window T"""

	model.eval()  # Put the model in test mode

	(inputs, labels) = next(iter(data_loader))

	# Transfer to GPU
	inputs = inputs.type(torch.FloatTensor).to(device)

	# forward pass to get layerwise activation values
	net_act_max = model.forward(inputs, isCalVnorm=True, percent=percent)

	return net_act_max