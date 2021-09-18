import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import copy

from lib.snn import LinearBN1d, ConvBN2d
from lib.functional import ZeroExpandInput

layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
stride_list = [1, 2, 2, 1, 2]


class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 3, stride=1, padding=1),
						nn.BatchNorm2d(96, eps=1e-4, momentum=0.9))

		self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 3, stride=2, padding=1),
						nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))

		self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, stride=2, padding=1),
						nn.BatchNorm2d(384, eps=1e-4, momentum=0.9))

		self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, stride=1, padding=1),
						nn.BatchNorm2d(384, eps=1e-4, momentum=0.9))	

		self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, stride=2, padding=1),
						nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))	

		self.fc6 = nn.Sequential(nn.Linear(4*4*256, 2048),
						nn.BatchNorm1d(2048, eps=1e-4, momentum=0.9))

		self.fc7 = nn.Linear(2048, 10)

	def forward(self, x, isCalVnorm=False, percent=99):
		x = x.view(-1, 3, 32, 32)

		# Conv Layer
		x1 = F.relu(self.conv1(x))
		x2 = F.relu(self.conv2(x1))
		x3 = F.relu(self.conv3(x2))
		x4 = F.relu(self.conv4(x3))
		x5 = F.relu(self.conv5(x4))

		# FC Layers
		x5 = x5.view(x5.size(0), -1)
		x6 = F.relu(self.fc6(x5))
		out = self.fc7(x6)

		if not isCalVnorm:
			return F.log_softmax(out, dim=1)
		else:
			net_act = [x1, x2, x3, x4, x5, x6] # record of intermediate layers' activation values
			net_act_top_percentile = [np.percentile(act.view(-1).cpu().detach().numpy(), percent) for act in net_act]

			return net_act_top_percentile


class sAlexNet(nn.Module):
	def __init__(self, model_old, Tencode, layer_list, replace_idx, conv_stride_list, vthr_list, neuronParam, device):
		super(sAlexNet, self).__init__()
		self.T = Tencode
		self.layer_list = layer_list
		self.replace_idx = replace_idx
		self.stride = conv_stride_list
		self.device = device
		self.vthr = vthr_list
		self.neuronParam = neuronParam
		self.conv1 = self._layer_init(model_old.conv1, 0)
		self.conv2 = self._layer_init(model_old.conv2, 1)
		self.conv3 = self._layer_init(model_old.conv3, 2)
		self.conv4 = self._layer_init(model_old.conv4, 3)
		self.conv5 = self._layer_init(model_old.conv5, 4)
		self.fc6 = self._layer_init(model_old.fc6, 5)
		self.fc7 = self._layer_init(model_old.fc7, 6)

	def _layer_init(self, layer, layer_idx):
		'''init network layer, and replace the ANN layer with the hybrid layer'''
		if layer_idx == self.replace_idx:
			if self.layer_list[layer_idx].startswith('conv'):
				out_channel, in_channel, kernel_size0, kernel_size1 = layer[0].weight.shape
				layer_updated = ConvBN2d(in_channel, out_channel, kernel_size0, stride=self.stride[layer_idx],\
										 padding=1, vthr=self.vthr[layer_idx], neuronParam=self.neuronParam, \
										 device=self.device)
				layer_updated.conv2d = copy.deepcopy(layer[0]) # copy weights from pre-trained ann layer
				layer_updated.bn2d = copy.deepcopy(layer[1]) # copy bn from pre-trained ann layer

			elif self.layer_list[layer_idx].startswith('fc'):
				layer_updated = LinearBN1d(layer[0].weight.shape[1], layer[0].weight.shape[0], \
										   neuronParam=self.neuronParam, vthr=self.vthr[layer_idx], \
										   device=self.device)
				layer_updated.linear = copy.deepcopy(layer[0]) # copy weights from pre-trained ann layer
				layer_updated.bn1d = copy.deepcopy(layer[1]) # copy bn from pre-trained ann layer
		# copy the original layer
		else:
			layer_updated = copy.deepcopy(layer)

		return layer_updated

	def forward(self, x, isCalVnorm=False, percent=99):
		x = x.view(-1, 3*32*32)
		x_spike, x = ZeroExpandInput.apply(x, self.T, self.device)
		x_spike = x_spike.view(-1, self.T, 3, 32, 32)
		x = x.view(-1, 3, 32, 32)

		if not isCalVnorm:
			x_spike, x, output = self.conv1(x_spike, x)
			for iLayer in range(1, len(self.layer_list)):
				if self.layer_list[iLayer] == 'fc6':
					x = x.view(x.size(0), -1)
					x_spike = x_spike.view(x_spike.size(0), self.T, -1)

				# forward pass in the hybrid network
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

					# apply activation function
					if iLayer == len(self.layer_list)-1:
						x = layer_ann(x)
					else:
						x = F.relu(layer_ann(x))

			return F.log_softmax(x, dim=1)
		else:
			net_act_top_percentile = []
			x_spike, x, x_ann = self.conv1(x_spike, x)
			net_act_top_percentile.append(
				np.percentile(x_ann.view(-1).cpu().detach().numpy(), percent))  # record ann layer activation value

			for iLayer in range(1, len(self.layer_list)):
				if self.layer_list[iLayer] == 'fc6':
					x = x.view(x.size(0), -1)
					x_spike = x_spike.view(x_spike.size(0), self.T, -1)

				# forward pass in the hybrid network
				if iLayer <= self.replace_idx:
					if self.layer_list[iLayer].startswith('conv'):
						layer_hybrid = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_hybrid = eval('self.fc'+str(iLayer+1))

					x_spike, x, x_ann = layer_hybrid(x_spike, x)
					net_act_top_percentile.append(np.percentile(x_ann.view(-1).cpu().detach().numpy(),percent))  # record ann layer activation value
				else:
					if self.layer_list[iLayer].startswith('conv'):
						layer_ann = eval('self.conv'+str(iLayer+1))
					elif self.layer_list[iLayer].startswith('fc'):
						layer_ann = eval('self.fc'+str(iLayer+1))

					# apply activation function
					if iLayer == len(self.layer_list)-1:
						x = layer_ann(x)
					else:
						x = F.relu(layer_ann(x))
					net_act_top_percentile.append(np.percentile(x.view(-1).cpu().detach().numpy(), percent))  # record ann layer activation value

			return net_act_top_percentile