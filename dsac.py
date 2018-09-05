import torch
import torch.nn.functional as F

import random

class DSAC:
	'''
	Differentiable RANSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function

	def __sample_hyp(self, x, y):
		'''
		Calculate a line hypothesis (slope, intercept) from two random points.

		x -- vector of x values
		y -- vector of y values
		'''

		# select two random points
		num_correspondences = x.size(0)

		idx1 = random.randint(0, num_correspondences-1)
		idx2 = random.randint(0, num_correspondences-1)

		tries = 1000

		# prevent slope from getting too large
		while torch.abs(x[idx1] - x[idx2]) < 0.01 and tries > 0:
			idx2 = random.randint(0, num_correspondences-1)
			tries = tries - 1

		if tries == 0: return 0, 0, False # no valid hypothesis found, indicated by False

		slope = (y[idx1] - y[idx2]) / (x[idx1] - x[idx2])
		intercept = y[idx1] - slope * x[idx1]

		return slope, intercept, True # True indicates success

	def __soft_inlier_count(self, slope, intercept, x, y):
		'''
		Soft inlier count for a given line and a given set of points.

		slope -- slope of the line
		intercept -- intercept of the line
		x -- vector of x values
		y -- vector of y values
		'''

		# point line distances
		dists = torch.abs(slope * x - y + intercept)
		dists = dists / torch.sqrt(slope * slope + 1)

		# soft inliers
		dists = 1 - torch.sigmoid(self.inlier_beta * (dists - self.inlier_thresh)) 
		score = torch.sum(dists)

		return score, dists

	def __refine_hyp(self, x, y, weights):
		'''
		Refinement by weighted Deming regression.

		Fits a line minimizing errors in x and y, implementation according to: 
			'Performance of Deming regression analysis in case of misspecified 
			analytical error ratio in method comparison studies'
			Kristian Linnet, in Clinical Chemistry, 1998

		x -- vector of x values
		y -- vector of y values
		weights -- vector of weights (1 per point)		
		'''

		ws = weights.sum()
		xm = (x * weights).sum() / ws
		ym = (y * weights).sum() / ws

		u = (x - xm)**2
		u = (u * weights).sum()

		q = (y - ym)**2
		q = (q * weights).sum()

		p = torch.mul(x - xm, y - ym)
		p = (p * weights).sum()

		slope = (q - u + torch.sqrt((u - q)**2 + 4*p*p)) / (2*p)
		intercept = ym - slope * xm

		return slope, intercept
		

	def __call__(self, prediction, labels):
		'''
		Perform robust, differentiable line fitting according to DSAC.

		Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of parameters (intercept, slope)
		'''

		# working on CPU because of many, small matrices
		prediction = prediction.cpu()

		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_top_loss = 0 # loss of best hypothesis

		self.est_parameters = torch.zeros(batch_size, 2) # estimated lines
		self.est_losses = torch.zeros(batch_size) # loss of estimated lines
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				slope, intercept, valid = self.__sample_hyp(x, y)
				if not valid: continue # skip invalid hyps

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self.__soft_inlier_count(slope, intercept, x, y)

				# === step 3: refine hypothesis ===========================
				slope, intercept = self.__refine_hyp(x, y, inliers)

				hyp = torch.zeros([2])
				hyp[1] = slope
				hyp[0] = intercept

				# === step 4: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_losses[b] = loss
					self.est_parameters[b] = hyp
					self.batch_inliers[b] = inliers

			# === step 5: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			exp_loss = torch.sum(hyp_losses * hyp_scores)
			avg_exp_loss = avg_exp_loss + exp_loss

			# loss of best hypothesis (for evaluation)
			avg_top_loss = avg_top_loss + self.est_losses[b]
	
		return avg_exp_loss / batch_size, avg_top_loss / batch_size
