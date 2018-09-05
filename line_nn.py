import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class LineNN(nn.Module):
	'''
	Genereric CNN architecture that can be used for 2D point prediction and direct prediction.

	It supports a FCN style architecture with varying receptive fields, as well as a globel 
	CNN which produces one output per image.

	'''

	def __init__(self, net_capacity, receptive_field = 0, direct = False, image_size = 64, global_output_grid = 8):
		'''
		Constructor.

		net_capacity -- multiplicative factor for the number of layer channels
		receptive field -- receptive field of the output neurons, the class will select 
			filter strides accordingly (supported: 15, 29, 37, 51, 65, 0), 0 = global 
			receptive field (default 0)
		direct -- model predicts line parameters directly, it predicts multiple 2D points 
			otherwise (default False)
		image_size -- size of the input images (default 64)
		global_output_grid -- number of 2D output points for a global model 
			(receptive_field=0), points are distributed on a 2D grid, i.e. number of 
			points is squared, for a receptive_field > 0 (i.e. FCN setting) the 
			number of output points results from the input image dimensions (default 8)
		'''
		super(LineNN, self).__init__()

		c = net_capacity
		output_dim = 2
		
		if direct and receptive_field is not 0:
			print('Warning: Direct models must have global receptive field (0).')

		# set the conv strides to achieve the desired receptive field
		self.global_model = False
		if receptive_field == 15:
			strides = [1, 1, 1, 1, 1, 1, 8]
		elif receptive_field == 29:
			strides = [1, 1, 1, 2, 2, 1, 2]
		elif receptive_field == 37:
			strides = [1, 1, 1, 2, 2, 2, 1]
		elif receptive_field == 51:
			strides = [1, 1, 2, 2, 2, 1, 1]
		elif receptive_field == 65:
			strides = [1, 2, 2, 2, 1, 1, 1]
		else:
			if receptive_field is not 0:
				print('Warning: Unknown receptive field, using 0 (global).')

			receptive_field = 2 * image_size # set global receptive field
			strides = [1, 2, 2, 2, 2, 2, 2]	
			if not direct: output_dim = global_output_grid * global_output_grid * 2
			self.global_model = True

		# build network
		self.conv1 = nn.Conv2d(3, 4*c, 3, strides[0], 1)
		self.bn1 = nn.BatchNorm2d(4*c)
		self.conv2 = nn.Conv2d(4*c, 8*c, 3, strides[1], 1)
		self.bn2 = nn.BatchNorm2d(8*c)
		self.conv3 = nn.Conv2d(8*c, 16*c, 3, strides[2], 1)
		self.bn3 = nn.BatchNorm2d(16*c)
		self.conv4 = nn.Conv2d(16*c, 32*c, 3, strides[3], 1)
		self.bn4 = nn.BatchNorm2d(32*c)
		self.conv5 = nn.Conv2d(32*c, 64*c, 3, strides[4], 1)
		self.bn5 = nn.BatchNorm2d(64*c)
		self.conv6 = nn.Conv2d(64*c, 64*c, 3, strides[5], 1)
		self.bn6 = nn.BatchNorm2d(64*c)
		self.conv7 = nn.Conv2d(64*c, 64*c, 3, strides[6], 1)
		self.bn7 = nn.BatchNorm2d(64*c)
		
		self.pool = nn.AdaptiveMaxPool2d(1) #used only for global models to support arbitrary image size

		self.fc1 = nn.Conv2d(64*c, 64*c, 1, 1, 0)
		self.bn8 = nn.BatchNorm2d(64*c)
		self.fc2 = nn.Conv2d(64*c, 64*c, 1, 1, 0)
		self.bn9 = nn.BatchNorm2d(64*c)
		self.fc3 = nn.Conv2d(64*c, output_dim, 1, 1, 0)

		self.patch_size = receptive_field / image_size
		self.global_output_grid = global_output_grid
		self.direct_model = direct

	def forward(self, input):
		'''
		Forward pass.

		input -- 4D data tensor (BxCxHxW)
		'''

		batch_size = input.size(0)

		x = F.relu(self.bn1(self.conv1(input)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.relu(self.bn6(self.conv6(x)))
		x = F.relu(self.bn7(self.conv7(x)))

		if self.global_model: x = self.pool(x)

		x = F.relu(self.bn8(self.fc1(x)))
		x = F.relu(self.bn9(self.fc2(x)))
		x = self.fc3(x)

		# direct model predicts line paramters directly
		if self.direct_model: 
			return x.squeeze()

		# otherwise points are predicted	
		x = torch.sigmoid(x) # normalize to 0,1

		if self.global_model: 
			x = x.view(-1, 2, self.global_output_grid, self.global_output_grid)

		# map local (patch-centric) point predictions to global image coordinates
		# i.e. distribute the points over the image
		patch_offset = 1 / x.size(2)

		x = x * self.patch_size - self.patch_size / 2 + patch_offset / 2

		for col in range(0, x.size(3)):
			x[:,1,:,col] = x[:,1,:,col] + col * patch_offset
			
		for row in range(0, x.size(2)):
			x[:,0,row,:] = x[:,0,row,:] + row * patch_offset

		return x.view(batch_size, 2, -1)
