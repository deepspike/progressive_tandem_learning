import torch
import time
import os 

from models.cifar10.AlexNet import layer_list, stride_list, AlexNet, sAlexNet
from data.data_loader_cifar10 import get_train_valid_loader, get_test_loader
from lib.classification import training, testing
from lib.convert import netUpdateAcc, vthrNorm
from lib.utils import dump_json

# Load datasets
home_dir = os.getcwd()
data_dir = os.path.join(home_dir, 'data/')
ann_ckp_dir = os.path.join(home_dir, 'exp/cifar10/')
assert os.path.exists(os.path.join(ann_ckp_dir, 'cifar10_alexNet_baseline.pt')), 'The checkpoint for the pre-trained ANN is not available.'
snn_ckp_dir = os.path.join(home_dir, 'exp/cifar10/AlexNetSNN/')

(train_loader, val_loader) = get_train_valid_loader(data_dir, batch_size=512)
test_loader = get_test_loader(data_dir, batch_size=512)

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	# Parameters
	Tencode = 16
	num_epochs = 150
	lr = 1e-3
	global best_acc_snn 
	best_acc_snn = 0
	Tpatient = 6 # patience period in epochs before a new layer is replaced
	t = 0 # patience counter
	k = 0 # replacing layer index
	l_total = len(layer_list) - 2 # the layer index of the last layer to replace
	test_acc_history = []
	test_loss_history = [] 
	epoch_convert = [0]
	vthr_convert = []
	neuronParam = {
		'neuronType': 'IF',
	}

	# Init Model and training configuration
	model = AlexNet()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
	criterion = torch.nn.CrossEntropyLoss()
	model = model.to(device)

	# Load Pre-trainde ANN model
	checkpoint = torch.load(os.path.join(ann_ckp_dir, 'cifar10_alexNet_baseline.pt'))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	best_acc_ann = 	checkpoint['acc']
	print('Accuracy of the Pre-trained AlexNet {}'.format(best_acc_ann))

	# replace the 1st layer
	print('-------------------- Replace Layer # ', k+1)
	# Analyse layerwise activation values and renormalize the threshold
	layer_act = vthrNorm(model, train_loader, device, percent=99.9)
	vthr_list = [act / Tencode for act in layer_act]
	vthr_convert.append(vthr_list[k])
	print("Updated neuron threshold of each layer ", vthr_convert)

	# replace the ann layer with a hybrid layer
	model = sAlexNet(model, Tencode, layer_list, k, stride_list, vthr_list, neuronParam, device)
	model = model.to(device)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
	print(model)
	t = 0 

	for epoch in range(num_epochs):
		since = time.time()
		
		# Training Stage
		model, acc_train, loss_train = training(model, train_loader, optimizer, criterion, device)

		# Validating stage
		acc_val, loss_val = testing(model, val_loader, criterion, device)

		# Testing Stage
		acc_test, loss_test = testing(model, test_loader, criterion, device)
		test_acc_history.append(acc_test)
		test_loss_history.append(loss_test)

		# Training Record
		time_elapsed = time.time() - since
		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60, time_elapsed % 60))
		print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
		print('Validation Accuracy: {:4f}, Loss: {:4f}'.format(acc_val, loss_val))
		print('Test Accuracy: {:4f}'.format(acc_test))

		# Update Model
		if k <= l_total:
			layer_act = vthrNorm(model, train_loader, device, percent=99.9)
			vthr_list = [act / Tencode for act in layer_act]
			model, optimizer, k, t, best_acc_snn, epoch_convert, vthr_convert = netUpdateAcc(model, optimizer, lr,
																							 acc_val, best_acc_snn, k,
																							 t, epoch, epoch_convert,
																							 Tpatient, layer_list,
																							 Tencode, snn_ckp_dir, device,
																							 sAlexNet, vthr_list, \
																							 neuronParam, vthr_convert,
																							 stride_list)
	# Save training record
	best_test_acc = max(test_acc_history[epoch_convert[-1]:])
	print('Test Accuracy of best SNN model {}'.format(best_test_acc))

	training_record = {
		'test_acc_history': test_acc_history,
		'test_loss_history': test_loss_history,
		'epoch_convert': epoch_convert,
		'best_snn_acc': best_test_acc,
		'best_ann_acc': best_acc_ann,
	}
	dump_json(training_record, snn_ckp_dir, 'spiking_alex_train_record')