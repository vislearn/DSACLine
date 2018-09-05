import torch

class LineLoss:
	'''
	Compares two lines by calculating the distance between their ends in the image.
	'''

	def __init__(self, image_size):
		'''
		Constructor.

		image_size -- size of the input images, used to normalize the loss
		'''
		self.image_size = image_size
	
	def __get_max_points(self, slope, intercept):
		'''
		Calculates the 2D points where a line intersects with the image borders.

		slope -- slope of the line
		intercept -- intercept of the line
		'''
		pts = torch.zeros([2, 2])

		x0 = 0
		x1 = 1
		y0 = intercept
		y1 = intercept + slope
		
		# determine which image borders the line cuts
		cuts_x0 = (y0 >= 0 and y0 <= 1)					# left border
		cuts_x1 = (y1 >= 0 and y1 <= 1)					# right border
		cuts_y0 = (y0 <= 0 and y1 >= 0) or (y1 <= 0 and y0 >= 0)	# upper border
		cuts_y1 = (y0 <= 1 and y1 >= 1) or (y1 <= 1 and y0 >= 1)	# lower border

		if cuts_x0 and cuts_x1:
			# line goes from left to right
			# use initialization above
			pass

		elif cuts_x0 and cuts_y0:
			# line goes from left to top
			y1 = 0
			x1 = -intercept / slope
		
		elif cuts_x0 and cuts_y1:
			# line goes from left to bottom
			y1 = 1
			x1 = (1 - intercept) / slope

		elif cuts_x1 and cuts_y0: 
			# line goes from top to right
			y0 = 0
			x0 = -intercept / slope

		elif cuts_x1 and cuts_y1: 
			# line goes from bottom to right
			y0 = 1
			x0 = (1 - intercept) / slope

		elif cuts_y0 and cuts_y1:
			# line goes from top to bottom
			y0 = 0
			x0 = -intercept / slope
			y1 = 1
			x1 = (1 - intercept) / slope

		else: 	
			# outside image
			x0 = -intercept / slope
			if abs(x0) < abs(y0): 
				y0 = 0
			else: 
				x0 = 0 

			x1 = (1 - intercept) / slope
			if abs(x1) < abs(y1):
				y1 = 1
			else: 
				x1 = 1 
	
		pts[0, 0] = x0
		pts[0, 1] = y0
		pts[1, 0] = x1
		pts[1, 1] = y1

		return pts

	def __call__(self, est, gt):
		'''
		Calculate the line loss.

		est -- estimated line, form: [intercept, slope]
		gt -- ground truth line, form: [intercept, slope]
		'''

		pts_est = self.__get_max_points(est[1], est[0])
		pts_gt = self.__get_max_points(gt[1], gt[0])

		# not clear which ends of the lines should be compared (there are ambigious cases), compute both and take min
		loss1 = pts_est - pts_gt
		loss1 = loss1.norm(2, 1).sum()

		flip_mat = torch.zeros([2, 2])
		flip_mat[0, 1] = 1
		flip_mat[1, 0] = 1

		loss2 = pts_est - flip_mat.mm(pts_gt)
		loss2 = loss2.norm(2, 1).sum()

		return min(loss1, loss2) * self.image_size
