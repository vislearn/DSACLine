import numpy
import cv2
import os

infolder = './images_rf65_c4_h64_t0.05_schedule4_2' 
outfolder = infolder + '_out' 

files = os.listdir(infolder)
files.sort()

for i,f in enumerate(files):

	infile = infolder + '/' + f

	img = cv2.imread(infile)

	img = cv2.flip(img, 0)
	img.resize((250, 600, 3))
	img = cv2.flip(img, 0)
	img.resize((300, 600, 3))

	iteration = i * 100
	label = 'learning iteration %05d/50000' % iteration
	cv2.putText(img, label, (330, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	label = 'validation inputs'
	cv2.putText(img, label, (30, 270), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	label = 'DSAC predictions'
	cv2.putText(img, label, (230, 270), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	label = 'direct predictions'
	cv2.putText(img, label, (420, 270), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	yoffset = 3

	cv2.line(img, (10, 5+yoffset), (20, 15+yoffset), (0, 255, 0))
	label = 'ground truth'
	cv2.putText(img, label, (25, 15+yoffset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	cv2.line(img, (10, 25+yoffset), (20, 35+yoffset), (255, 0, 0))
	label = 'predicted line'
	cv2.putText(img, label, (25, 35+yoffset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	cv2.rectangle(img, (170, 5+yoffset), (180, 15+yoffset), (0, 0, 254))
	label = 'incorrect'
	cv2.putText(img, label, (185, 15+yoffset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	cv2.rectangle(img, (170, 25+yoffset), (180, 35+yoffset), (0, 255, 0))
	label = 'correct'
	cv2.putText(img, label, (185, 35+yoffset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

	if not os.path.exists(outfolder):
		os.mkdir(outfolder)

	outfile = (outfolder + '/out_%06d.png') % i

	cv2.imwrite(outfile, img)
