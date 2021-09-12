import torch.nn as nn
import torch.nn.functional as F
import copy 
import numpy as np

from lib.snn import LinearBN1d, ConvBN2d
from lib.functional import ZeroExpandInput

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc9', 'fc10', 'fc11']
stride_list = [1, 2, 1, 2, 1, 2, 1, 2]


class VGG11(nn.Module):
	def __init__(self):
		super(VGG11, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
						nn.BatchNorm2d(64, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
						nn.BatchNorm2d(128, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
						nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
						nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))	

		self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.conv7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))	

		self.conv8 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
						nn.BatchNorm2d(512, eps=1e-4, momentum=0.9),
						nn.ReLU(inplace=True))

		self.fc9 = nn.Sequential(nn.Linear(2*2*512, 4096),
					nn.BatchNorm1d(4096, eps=1e-4, momentum=0.9))

		self.fc10 = nn.Sequential(nn.Linear(4096, 4096),
					nn.BatchNorm1d(4096, eps=1e-4, momentum=0.9))

		self.fc11 = nn.Linear(4096, 10)

	def forward(self, x, isCalVnorm=False, percent=99):
		x = x.view(-1, 3, 32, 32)

		# Conv Layer
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)
		x6 = self.conv6(x5)
		x7 = self.conv7(x6)
		x8 = self.conv8(x7)

		# FC Layers
		x8 = x8.view(x8.size(0), -1)
		x9 = F.relu(self.fc9(x8))
		x10 = F.relu(self.fc10(F.dropout(x9)))
		out = self.fc11(F.dropout(x10))

		if not isCalVnorm:
			return F.log_softmax(out, dim=1)
		else:
			net_act = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] # record of intermediate layers' activation values
			net_act_top_percentile = [np.percentile(act.view(-1).cpu().detach().numpy(), percent) for act in net_act]

			return net_act_top_percentile


class sVGG11(nn.Module):
	''' build hybrid network '''

	def __init__(self, model_old, Tsim, layer_list, replace_idx, stride_list, vthr_list, neuronParam, device):
		super(sVGG11, self).__init__()
		self.T = Tsim
		self.layer_list = layer_list
		self.replace_idx = replace_idx
		self.num_layers = len(layer_list)
		self.stride = stride_list
		self.device = device
		self.vthr = vthr_list
		self.neuronParam = neuronParam
		self.conv1 = self._layer_init(model_old.conv1, 0)
		self.conv2 = self._layer_init(model_old.conv2, 1)
		self.conv3 = self._layer_init(model_old.conv3, 2)
		self.conv4 = self._layer_init(model_old.conv4, 3)
		self.conv5 = self._layer_init(model_old.conv5, 4)
		self.conv6 = self._layer_init(model_old.conv6, 5)
		self.conv7 = self._layer_init(model_old.conv7, 6)
		self.conv8 = self._layer_init(model_old.conv8, 7)
		self.fc9 = self._layer_init(model_old.fc9, 8)
		self.fc10 = self._layer_init(model_old.fc10, 9)
		self.fc11 = self._layer_init(model_old.fc11, 10)

	def _layer_init(self, layer, layer_idx):
		# replace the ANN layer with hybrid layer
		if layer_idx == self.replace_idx:
			if self.layer_list[layer_idx].startswith('conv'):
				out_channel, in_channel, kernel_size, _ = layer[0].weight.shape
				layer_updated = ConvBN2d(in_channel, out_channel, kernel_size, stride=self.stride[layer_idx], \
										 padding=1, vthr=self.vthr[layer_idx], neuronParam=self.neuronParam,\
										 device=self.device)
				layer_updated.conv2d = copy.deepcopy(layer[0]) # copy weights from pre-trained ann layer
				layer_updated.bn2d = copy.deepcopy(layer[1]) # copy bn from pre-trained ann layer

			elif self.layer_list[layer_idx].startswith('fc'):
				layer_updated = LinearBN1d(layer[0].weight.shape[1], layer[0].weight.shape[0],\
										   vthr=self.vthr[layer_idx], neuronParam=self.neuronParam,\
										   device=self.device)
				layer_updated.linear = copy.deepcopy(layer[0]) # copy weights from pre-trained ann layer
				layer_updated.bn1d = copy.deepcopy(layer[1]) # copy bn from pre-trained ann layer

		# keep the original ANN layer
		else:
			layer_updated = copy.deepcopy(layer)

		return layer_updated

	def forward(self, x, isCalVnorm=False, percent=99):
		x = x.view(-1, 3*32*32)
		x_spike, x = ZeroExpandInput.apply(x, self.T, self.device)
		x_spike = x_spike.view(-1, self.T, 3, 32, 32)
		x = x.view(-1, 3, 32, 32)

		if not isCalVnorm:
			x_spike, x, x_ann = self.conv1(x_spike, x)
			for iLayer in range(1, self.num_layers):
				# expansion of conv feature maps
				if self.layer_list[iLayer] == 'fc9':
					x = x.view(x.size(0), -1)
					x_spike = x_spike.view(x_spike.size(0), self.T, -1)

				if iLayer <= self.replace_idx:
					if self.layer_list[iLayer].startswith('conv'):
						layer_hybrid = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_hybrid = eval('self.fc'+str(iLayer+1))

					x_spike, x, x_ann = layer_hybrid(x_spike, x)
				else:
					if self.layer_list[iLayer].startswith('conv'):
						layer_ann = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_ann = eval('self.fc'+str(iLayer+1))

					if self.layer_list[iLayer] == self.layer_list[-1]:
						x = layer_ann(x) # output layer
					else:
						x = F.relu(layer_ann(x))

			return F.log_softmax(x, dim=1)
		else:
			net_act_top_percentile = []
			x_spike, x, x_ann = self.conv1(x_spike, x)
			net_act_top_percentile.append(
				np.percentile(x_ann.view(-1).cpu().detach().numpy(), percent))  # record ann layer activation value

			for iLayer in range(1, self.num_layers):
				# expansion of conv feature maps
				if self.layer_list[iLayer] == 'fc9':
					x = x.view(x.size(0), -1)
					x_spike = x_spike.view(x_spike.size(0), self.T, -1)

				if iLayer <= self.replace_idx:
					if self.layer_list[iLayer].startswith('conv'):
						layer_hybrid = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_hybrid = eval('self.fc'+str(iLayer+1))

					x_spike, x, x_ann = layer_hybrid(x_spike, x)
					net_act_top_percentile.append(np.percentile(x_ann.view(-1).cpu().detach().numpy(),
																percent))  # record ann layer activation value
				else:
					if self.layer_list[iLayer].startswith('conv'):
						layer_ann = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_ann = eval('self.fc'+str(iLayer+1))

					if self.layer_list[iLayer] == self.layer_list[-1]:
						x = layer_ann(x) # output layer
					else:
						x = F.relu(layer_ann(x))
					net_act_top_percentile.append(np.percentile(x_ann.view(-1).cpu().detach().numpy(),
																percent))  # record ann layer activation value

			return net_act_top_percentile