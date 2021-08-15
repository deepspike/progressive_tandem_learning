import time
import os 
import torch

from models.cifar10.vgg11 import VGG11
from data.data_loader_cifar10 import get_train_valid_loader, get_test_loader
from lib.classification import training, testing

# Load datasets
home_dir = os.getcwd()
data_dir = os.path.join(home_dir, 'data/')
ckp_dir = os.path.join(home_dir, 'exp/cifar10/')

(train_loader, val_loader) = get_train_valid_loader(data_dir)
test_loader = get_test_loader(data_dir)

if __name__ == '__main__':        
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	# Parameters
	num_epochs = 100
	global best_acc 
	best_acc = 0

	# Models and training configuration 
	model = VGG11()
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
	criterion = torch.nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		since = time.time()
		
		# Training Stage
		model, acc_train, loss_train = training(model, train_loader, optimizer, criterion, device)

		# Validating stage
		acc_val, loss_val = testing(model, val_loader, criterion, device)

		# Testing Stage
		acc_test, loss_test = testing(model, test_loader, criterion, device)

		# Training Record
		time_elapsed = time.time() - since
		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60, time_elapsed % 60))
		print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
		print('Validation Accuracy: {:4f}, Loss: {:4f}'.format(acc_val, loss_val))
		print('Test Accuracy: {:4f}'.format(acc_test))

		# Save Model
		if acc_val > best_acc:
			print("Saving the model.")\

			if not os.path.exists(ckp_dir):
				os.makedirs(ckp_dir)

			state = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss_train,
					'acc': acc_test,
			}
			torch.save(state, ckp_dir+'cifar10_vgg11_baseline.pt')
			best_acc = acc_val
		