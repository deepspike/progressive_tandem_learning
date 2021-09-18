import torch
import torch.nn.functional as F

def IF(x, mem, spike, vthr):
	""" integrate-and-fire Neuron Model """

	mem = mem + x - spike * vthr
	spike = torch.ge(mem, vthr).float()

	return mem, spike


class LinearIF(torch.autograd.Function):
	"""Fully-connected SNN layer"""
	@staticmethod
	def forward(ctx, spike_in, ann_output, weight, device=torch.device('cuda'), bias=None, vthr=1.0, neuronParam=None):
		"""
		Params:
			spike_in: input spike trains
			ann_output: placeholder
			weight: connection weights
			device: cpu or cuda
			bias: neuronal bias parameters
			neuronParam： neuronal parameters
		Returns:
			spike_out: output spike trains
			spike_count_out: output spike counts
		"""
		supported_neuron = {
			'IF': IF,
		}
		if neuronParam['neuronType'] not in supported_neuron:
			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
		N, T, _ = spike_in.shape
		out_features = bias.shape[0]
		pot_in = spike_in.matmul(weight.t())
		spike_out = torch.zeros_like(pot_in, device=device)
		bias_distribute = bias.repeat(N, 1) / T  # distribute bias through simulation time steps
		mem = torch.zeros(N, out_features, device=device)
		spike = torch.zeros(N, out_features, device=device)  # init input spike train

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			x = pot_in[:, t, :].squeeze() + bias_distribute
			# Membrane potential update
			mem, spike = IF(x, mem, spike, vthr)
			spike_out[:, t, :] = spike

		spike_count_out = torch.sum(spike_out, dim=1).squeeze(dim=1)

		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""
		grad_ann_out = grad_spike_count_out.clone()

		return None, grad_ann_out, None, None, None, None, None


class Conv2dIF(torch.autograd.Function):
	"""2D Convolutional Layer"""

	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device('cuda'), bias=None, stride=1, padding=0, vthr=1.0, neuronParam=None):
		"""
		Params:
			spike_in: input spike trains
			features_in: placeholder
			weight: connection weights
			device: cpu or cuda
			bias: neuronal bias parameters
			stride: stride of 1D Conv
			padding: padding of 1D Conv
			dilation: dilation of 1D Conv
			neuronParam： neuronal parameters
		Returns:
			spike_out: output spike trains
			spike_count_out: output spike counts
		"""
		supported_neuron = {
			'IF': IF,
		}
		if neuronParam['neuronType'] not in supported_neuron:
			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
		N, T, in_channels, iH, iW = spike_in.shape
		out_channels, in_channels, kH, kW = weight.shape
		mem = torch.zeros_like(F.conv2d(spike_in[:, 0, :, :, :], weight, bias, stride, padding))
		bias_distribute = F.conv2d(torch.zeros_like(spike_in[:, 0, :, :, :]), weight, bias, stride, padding) / T
		_, _, outH, outW = mem.shape
		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
		spike = torch.zeros(N, out_channels, outH, outW, device=device)  # init input spike train

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			x = F.conv2d(spike_in[:, t, :, :, :], weight, None, stride, padding) + bias_distribute
			# Membrane potential update
			mem, spike = IF(x, mem, spike, vthr)
			spike_out[:, t, :, :] = spike

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()

		return None, grad_spike_count_out, None, None, None, None, None, None, None


class ZeroExpandInput(torch.autograd.Function):
	"""Zero-expand the input image along the time dimension"""
	@staticmethod
	def forward(ctx, input_image, T, device):
		"""
		Args:
			input_image: normalized within (0,1)
			T: time window size
			device: cpu or cuda
		"""
		N, dim = input_image.shape
		input_image_sc = input_image
		zero_inputs = torch.zeros(N, T-1, dim).to(device)
		input_image = input_image.unsqueeze(dim=1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		return input_image_spike, input_image_sc

	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None

