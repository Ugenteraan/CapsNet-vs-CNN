import os
import glob
from random import shuffle

#to keep track of files
number_of_files = 0

#get the current terminal's path
current_directory = os.getcwd()

#path where the datasets are stored
dataset_path = current_directory + '/datasets/'

#get the length to extract the name of the folder (i.e. styles) later on
dataset_path_length = len(dataset_path)

#determine whether the input images are coloured or grayscaled
grayscale = True

#set the depth of the image (grayscale = 1 channel only, coloured = 3 channels)
image_depth = 3 if grayscale is False else 1

#size of the image (height and width)
picture_input_dimension = 100
X_shape_digitCaps = 36992

#training iteration
epoch = 100

#this is to break the datasets in 'n' sizes 
memory_limit = 4

#stochastic gradient descent
batch_size = 1
accuracy_batch_size = 1

#learning rate of the model
learning_rate = 0.000001

#names of the folder to save and load models
save_folder_name = 'ckpt_folder'
load_folder_name = 'ckpt_folder'

#in list format in case there are other image extensions in the future
image_extension = ['.jpg']

#list to keep track of the name of the folder (i.e. styles)
folders = []

#names of all the files
filenames = []

regularization_scale = 0.392

epsilon = 1e-9

m_plus = 0.9

m_minus = 0.1

lambda_val = 0.5

routing_iter = 3
#function to check the extension of the file using the 'image_extension' variable earlier
def check_file_type(file_path):
	#one-liner code that returns true if the file's extension ends with '.jpg', returns false if not
	return True if file_path[-4:] in image_extension else False
	


#for all the folders (i.e. styles) in the directory, append to the list
for folder in glob.glob(dataset_path + '*'):
	#append the folder's name into the list
	#this is where the dataset_path_length variable declared earlier is used
	folders.append(folder[dataset_path_length:])


#sort the list alphabetically
folders = sorted(folders, key=str.lower)


#to iterate through all the data files and keep count of the number of files
folder_count = 0
for folder in folders:

	for file in glob.glob(dataset_path + folder + '/**', recursive=True):

		#only true if the file is indeed a file that ends with a '.jpg' extension
		if check_file_type(file):
			#increase the count of the variable everytime there's an image file
			number_of_files += 1
			filenames.append([file, folder_count])

	folder_count += 1

#randomize the list
shuffle(filenames)

print("CLASSES : ", folders)
#number of classes = number of folders
num_of_classes = len(folders)

print("TOTAL NUMBER OF FILES : ", number_of_files)