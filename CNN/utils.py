import cv2
import glob
import settings
import numpy as np 
from sklearn.model_selection import train_test_split


def get_files(number_of_round):

	#allX is a numpy array to store images
	#for grayscale input
	if settings.grayscale is True:
		#for grayscaled input, there is no dimension at the end
		allX = np.zeros((settings.number_of_files//settings.memory_limit + 1, settings.picture_input_dimension, settings.picture_input_dimension), dtype='float64')

	#for coloured input
	else:
		#for coloured input, the dimension of the numpy array is 3 at the end. 3 represents the 3 channels (RGB)
		allX = np.zeros((settings.number_of_files//settings.memory_limit + 1, settings.picture_input_dimension, settings.picture_input_dimension, 3), dtype='float64')

	#numpy array to store labels
	allY = np.zeros((settings.number_of_files//settings.memory_limit + 1))

	count = 0

	number_of_limit = settings.number_of_files // settings.memory_limit

	#starting value must be 0 at first round and then increases to the number_of_limit and so on
	#end value must have +1 at the back to iterate through all the files
	for memrange in range( (number_of_round - 1 ) * number_of_limit , number_of_round * (number_of_limit) + 1):

		try:
			if settings.grayscale is True:
				img = cv2.imread(settings.filenames[memrange][0], 0)
			else:
				img = cv2.imread(settings.filenames[memrange][0])

			resized_img = cv2.resize(img, (settings.picture_input_dimension, settings.picture_input_dimension))

			allX[count] = np.array(resized_img)

			allY[count] = settings.filenames[memrange][1]

			count += 1


		except IndexError:

			print("Index out of bounds!")





	#split the data into training set and test set and randomize them (order of corresponding allX and allY is preserved)
	X, X_test, Y, Y_test = train_test_split(allX, allY, test_size = 0.2, random_state=42)
	allX, allY = None, None
	return(X,X_test,Y,Y_test)
