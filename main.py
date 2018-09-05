import torch
import torch.optim as optim
import torchvision.utils as vutils

import os
import time
import numpy
import warnings
import argparse

from skimage.io import imsave

from line_dataset import LineDataset
from line_nn import LineNN
from line_loss import LineLoss

from dsac import DSAC

parser = argparse.ArgumentParser(description='This script creates a toy problem of fitting line parameters (slope+intercept) to synthetic images showing line segments, noise and distracting circles. Two networks are trained in parallel and compared: DirectNN predicts the line parameters directly (two output neurons). PointNN predicts a number of 2D points to which the line parameters are subsequently fitted using differentiable RANSAC (DSAC). The script will produce a sequence of images that illustrate the training process for both networks.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--receptivefield', '-rf', type=int, default=65, choices=[65, 51, 37, 29, 15, 0],
	help='receptive field size of the PointNN, i.e. one point prediction is made for each image patch of this size, different receptive fields are achieved by different striding strategies, 0 means global, i.e. the full image, the DirectNN will always use 0 (global)')

parser.add_argument('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of both networks (PointNN and DirectNN), it is a multiplicative factor for the number of channels in each network layer')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count. Its measured in relative image size (1 = image width)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.5, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--learningrate', '-lr', type=float, default=0.001, 
	help='learning rate')

parser.add_argument('--lrstep', '-lrs', type=int, default=2500, 
	help='cut learning rate in half each x iterations')

parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000, 
	help='keep initial learning rate for at least x iterations')

parser.add_argument('--batchsize', '-bs', type=int, default=32, 
	help='training batch size')

parser.add_argument('--trainiterations', '-ti', type=int, default=50000, 
	help='number of training iterations (= parameter updates)')

parser.add_argument('--imagesize', '-is', type=int, default=64, 
	help='size of input images generated, images are square')

parser.add_argument('--storeinterval', '-si', type=int, default=1000, 
	help='store network weights and a prediction vizualisation every x training iterations')

parser.add_argument('--valsize', '-vs', type=int, default=9, 
	help='number of validation images used to vizualize predictions')

parser.add_argument('--valthresh', '-vt', type=float, default=5, 
	help='threshold on the line loss for vizualizing correctness of predictions')

parser.add_argument('--cpu', '-cpu', action='store_true',
	help='execute networks on CPU. Note that (RANSAC) line fitting anyway runs on CPU')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files. Useful to separate different runs of the program')

opt = parser.parse_args()

if len(opt.session) > 0: opt.session = '_' + opt.session
sid = 'rf%d_c%d_h%d_t%.2f%s' % (opt.receptivefield, opt.capacity, opt.hypotheses, opt.inlierthreshold, opt.session)

# setup the training process
dataset = LineDataset(opt.imagesize, opt.imagesize)

loss = LineLoss(opt.imagesize)
dsac = DSAC(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, loss)

# we train two CNNs in parallel
# 1) a CNN that predicts points and is trained with DSAC -> PointNN (good idea)
point_nn = LineNN(opt.capacity, opt.receptivefield)
if  not opt.cpu: point_nn = point_nn.cuda()
point_nn.train()
opt_point_nn = optim.Adam(point_nn.parameters(), lr=opt.learningrate)
lrs_point_nn = optim.lr_scheduler.StepLR(opt_point_nn, opt.lrstep, gamma=0.5)

# 2) a CNN that predicts the line parameters directly -> DirectNN (bad idea)
direct_nn = LineNN(opt.capacity, 0, True)
if not opt.cpu: direct_nn = direct_nn.cuda()
direct_nn.train()
opt_direct_nn = optim.Adam(direct_nn.parameters(), lr=opt.learningrate)
lrs_direct_nn = optim.lr_scheduler.StepLR(opt_direct_nn, opt.lrstep, gamma=0.5)

# keep track of training progress
train_log = open('log_'+sid+'.txt', 'w', 1)

# some helper functions
def prepare_data(inputs, labels):
	# convert from numpy images to normalized torch arrays

	inputs = torch.from_numpy(inputs)
	labels = torch.from_numpy(labels)	

	if not opt.cpu: inputs = inputs.cuda()
	inputs.transpose_(1,3).transpose_(2, 3)
	inputs = inputs - 0.5 # normalization

	return inputs, labels

def batch_loss(prediction, labels):
	# caluclate the loss for each image in the batch

	losses = torch.zeros(labels.size(0))

	for b in range(0, labels.size(0)):
		losses[b] = loss(prediction[b], labels[b])

	return losses

# generate validation data (for consistent vizualisation only)
val_images, val_labels = dataset.sample_lines(opt.valsize)
val_inputs, val_labels = prepare_data(val_images, val_labels)

# start training
for iteration in range(0, opt.trainiterations+1):

	start_time = time.time()

	# generate training data
	inputs, labels = dataset.sample_lines(opt.batchsize)
	inputs, labels = prepare_data(inputs, labels)

	# point nn forward pass
	point_prediction = point_nn(inputs)

	# robust line fitting with DSAC
	exp_loss, top_loss = dsac(point_prediction, labels)
	
	exp_loss.backward()			# calculate gradients (pytorch autograd)
	opt_point_nn.step()			# update parameters 
	opt_point_nn.zero_grad()		# reset gradient buffer
	if iteration >= opt.lrstepoffset:	
		lrs_point_nn.step()		# update learning rate schedule

	# also train direct nn
	direct_prediction = direct_nn(inputs)
	direct_loss = batch_loss(direct_prediction, labels).mean()

	direct_loss.backward()			# calculate gradients (pytorch autograd)
	opt_direct_nn.step()			# update parameters 
	opt_direct_nn.zero_grad()		# reset gradient buffer
	if iteration >= opt.lrstepoffset: 
		lrs_direct_nn.step()		# update learning rate schedule

	# wrap up
	end_time = time.time()-start_time
	print('Iteration: %6d, DSAC Expected Loss: %2.2f, DSAC Top Loss: %2.2f, Direct Loss: %2.2f, Time: %.2fs' 
		% (iteration, exp_loss, top_loss, direct_loss, end_time), flush=True)

	train_log.write('%d %f %f %f\n' % (iteration, exp_loss, top_loss, direct_loss))

	del exp_loss, top_loss, direct_loss

	# store prediction vizualization and nn weights (each couple of iterations)
	if iteration % opt.storeinterval == 0:

		point_nn.eval()
		direct_nn.eval()

		# DSAC validation prediction
		prediction = point_nn(val_inputs)
		val_exp, val_loss = dsac(prediction, val_labels)
		val_correct = dsac.est_losses < opt.valthresh

		# direct nn validation prediction
		direct_val_est = direct_nn(val_inputs)
		direct_val_loss = batch_loss(direct_val_est, val_labels)
		direct_val_correct = direct_val_loss < opt.valthresh

		direct_val_est = direct_val_est.detach().cpu().numpy()
		dsac_val_est = dsac.est_parameters.detach().cpu().numpy()
		points = prediction.detach().cpu().numpy()

		# draw DSAC estimates
		viz_dsac = dataset.draw_models(val_labels)
		viz_dsac = dataset.draw_points(points, viz_dsac, dsac.batch_inliers)
		viz_dsac = dataset.draw_models(dsac_val_est, viz_dsac, val_correct)

		# draw direct estimates
		viz_direct = dataset.draw_models(val_labels)
		viz_direct = dataset.draw_models(direct_val_est, viz_direct, direct_val_correct)

		def make_grid(batch):
			batch = torch.from_numpy(batch)
			batch.transpose_(1, 3).transpose_(2, 3)		
			return vutils.make_grid(batch, nrow=3,normalize=False)

		viz_inputs = make_grid(val_images)
		viz_dsac = make_grid(viz_dsac)
		viz_direct = make_grid(viz_direct)

		viz = torch.cat((viz_inputs, viz_dsac, viz_direct), 2)
		viz.transpose_(0, 1).transpose_(1, 2)	
		viz = viz.numpy()
	
		# store image (and ignore warning about loss of precision)	
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			outfolder = 'images_' + sid
			if not os.path.isdir(outfolder): os.mkdir(outfolder)
			imsave('./%s/prediction_%s_%06d.png' % (outfolder, sid, iteration), viz)
	
		# store model weights
		torch.save(point_nn.state_dict(), './weights_pointnn_' + sid + '.net')
		torch.save(direct_nn.state_dict(), './weights_directnn_' + sid + '.net')

		print('Storing snapshot. Validation loss: %2.2f'% val_loss, flush=True)

		del val_exp, val_loss, direct_val_loss

		point_nn.train()
		direct_nn.train()

print('Done without errors.')
train_log.close()
