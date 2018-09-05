import random
import numpy as np
import math

from skimage.draw import line, line_aa, circle, set_color, circle_perimeter_aa
from skimage.io import imsave
from skimage.util import random_noise

maxSlope = 10 # restrict the maximum slope of generated lines for stability
minLength = 20 # restrict the minimum length of line segments

class LineDataset:
	'''
	Generator of line segment images.

	Images will have 1 random line segment each, filled with noise and distractor circles.
	Class also offers functionality for drawing line parameters, hypotheses and point predictions.
	'''

	def __init__(self, imgW = 64, imgH = 64, margin = -5, bg_clr = 0.5):
		'''
		Constructor. 

		imgW -- image width (default 64)
		imgH -- image height (default 64)
		margin -- lines segments are sampled within this margin, negative value means that a line segment can start or end outside the image (default -5)
		bg_clr -- background intensity (default 0.5)
		'''
		
		self.imgW = imgW
		self.imgH = imgH
		self.margin = margin
		self.bg_clr = bg_clr

	def draw_line(self, data, lX1, lY1, lX2, lY2, clr, alpha=1.0):
		'''
		Draw a line with the given color and opacity.

		data -- image to draw to
		lX1 -- x value of line segment start point
		lY1 -- y value of line segment start point
		lX2 -- x value of line segment end point
		lY2 -- y value of line segment end point
		clr -- line color, triple of values
		alpha -- opacity (default 1.0)
		'''

		rr, cc, val = line_aa(lY1, lX1, lY2, lX2)
		set_color(data, (rr, cc), clr, val*alpha)

	def draw_hyps(self, labels, scores, data=None):
		'''
		Draw a set of line hypothesis for a batch of images.
		
		labels -- line parameters, array shape (NxMx2) where 
			N is the number of images in the batch
			M is the number of hypotheses per image
			2 is the number of line parameters (intercept, slope)
		scores -- hypotheses scores, array shape (NxM), see above, higher score will be drawn with higher opacity
		data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels
		
		'''

		n = labels.shape[0] # number of images
		m = labels.shape[1] # number of hypotheses

		if data is None: # create new batch of images
			data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
			data.fill(self.bg_clr)

		clr = (0, 0, 1)

		for i in range (0, n):
			for j in range (0, m):
				lY1 = int(labels[i, j, 0] * self.imgH)
				lY2 = int(labels[i, j, 1] * self.imgW + labels[i, j, 0] * self.imgH)
				self.draw_line(data[i], 0, lY1, self.imgW, lY2, clr, scores[i, j])

		return data

	def draw_models(self, labels, data=None, correct=None):
		'''
		Draw lines for a batch of images.
	
		labels -- line parameters, array shape (Nx2) where 
			N is the number of images in the batch
			2 is the number of line parameters (intercept, slope)
		data -- batch of images to draw to, if empty a new batch wil be created according to the shape of labels 
			and lines will be green, lines will be blue otherwise
		correct -- array of shape (N) indicating whether a line estimate is correct 
		'''

		n = labels.shape[0]
		if data is None: 
			data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
			data.fill(self.bg_clr)
			clr = (0, 1, 0)
		else:
			clr = (0, 0, 1)

		for i in range (0, n):
			lY1 = int(labels[i, 0] * self.imgH)
			lY2 = int(labels[i, 1] * self.imgW + labels[i, 0] * self.imgH)
			self.draw_line(data[i], 0, lY1, self.imgW, lY2, clr)

			if correct is not None:
				
				# draw border green if estiamte is correct, red otherwise
				if correct[i]: borderclr = (0, 1, 0)
				else: borderclr = (1, 0, 0)
				
				set_color(data[i], line(0, 0, 0, self.imgW-1), borderclr)			
				set_color(data[i], line(0, 0, self.imgH-1, 0), borderclr)			
				set_color(data[i], line(self.imgH-1, 0, self.imgH-1, self.imgW-1), borderclr)			
				set_color(data[i], line(0, self.imgW-1, self.imgH-1, self.imgW-1), borderclr)			

		return data

	def draw_points(self, points, data, inliers=None):
		'''
		Draw 2D points for a batch of images.

		points -- 2D points, array shape (Nx2xM) where 
			N is the number of images in the batch
			2 is the number of point dimensions (x, y)
			M is the number of points
		data -- batch of images to draw to
		inliers -- soft inlier score for each point, 
			if given and score < 0.5 point will be drawn green, red otherwise
		'''

		n = points.shape[0] # number of images
		m = points.shape[2] # number of points

		for i in range (0, n):
			for j in range(0, m):

				clr = (0.2, 0.2, 0.2) # draw predicted points as dark circles
				if inliers is not None and inliers[i, j] > 0.5:
					clr = (0.7, 0.7, 0.7) # draw inliers as light circles
					
				r = int(points[i, 0, j] * self.imgH)
				c = int(points[i, 1, j] * self.imgW)
				rr, cc = circle(r, c, 2)
				set_color(data[i], (rr, cc), clr)

		return data

	def sample_lines(self, n):
		'''
		Create new input images of random line segments and distractors along with ground truth parameters.

		n -- number of images to create
		'''

		data = np.zeros((n, self.imgH, self.imgW, 3), dtype=np.float32)
		data.fill(self.bg_clr)
		labels = np.ndarray((n, 2, 1, 1), dtype=np.float32)

		for i in range (0, n): # for each image

			# create a random number of distractor circles
			nC = random.randint(2, 5)
			for c in range(0, nC):

				cR = random.randint(int(0.1 * self.imgW), int(1 * self.imgW))			
				cX1 = random.randint(int(-0.5 * cR), int(self.imgW+0.5*cR+1))
				cY1 = random.randint(int(-0.5 * cR), int(self.imgH+0.5*cR+1))	

				clr = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

				rr, cc, val =  circle_perimeter_aa(cY1, cX1, cR)
				set_color(data[i], (rr, cc), clr, val)

			# create line segment
			while True:

				# sample segment end points
				lX1 = random.randint(self.margin, self.imgW-self.margin+1)
				lX2 = random.randint(self.margin, self.imgW-self.margin+1)
				lY1 = random.randint(self.margin, self.imgH-self.margin+1)
				lY2 = random.randint(self.margin, self.imgH-self.margin+1)

				# check min length
				length = math.sqrt((lX1 - lX2) * (lX1 - lX2) + (lY1 - lY2) * (lY1 - lY2))
				if length < minLength: continue

				# random color
				clr = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

				# calculate line ground truth parameters
				delta = lX2 - lX1
				if delta == 0: delta = 1 

				slope = (lY2 - lY1) / delta
				intercept = lY1 - slope * lX1

				# not too steep for stability
				if abs(slope) < maxSlope: break

			labels[i, 0] = intercept / self.imgH
			labels[i, 1] = slope

			self.draw_line(data[i], lX1, lY1, lX2, lY2, clr)

			# apply some noise on top
			data[i] = random_noise(data[i], mode='speckle')

		return data, labels
