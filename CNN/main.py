import os
import utils
import model 
import settings
import numpy as np 
import tensorflow as tf 


#path to save the trained model
save_path = settings.current_directory + '/'+ settings.save_folder_name + '/model.ckpt'

#tensorboard initialization
# writer = tf.summary.FileWriter(settings.current_directory + '/tensorboard/1')

	
#initialize session
sess = tf.InteractiveSession()
#initialize the model file
model_vgg = model.Model()
#run the tensorflow session
sess.run(tf.global_variables_initializer())
#initialize the tensorflow saver module
saver = tf.train.Saver(model_vgg.wb_variables)

#exception handling in loading the model
try:
	#restore the saved model
	saver.restore(sess, settings.current_directory + '/'+settings.load_folder_name+'/model.ckpt')
	print("Model has been loaded !")
except:
	print("Model is not loaded !")

# writer.add_graph(sess.graph)


#function to perform the training in tensorflow
def training_session(training_dataset, training_labelset, testing_labelset, epoch):
	
	#initialize this variable to 0 everytime the function runs
	#this variable is used to add all the training accuracies and be averaged
	training_accuracy = 0

	#initialize this variable to 0 everytime the function runs
	#this variable is used as the last index of a particular batch size
	end_batch_size = 0

	#VARIABLE TO KEEP COUNT HOW MANY TIMES DID A CERTAIN NUMBER OF BATCH WENT IN FOR ONE EPOCH
	#TO CALCULATE THE ACCURACY 
	counter = 0

	#LOOP THAT ITERATES IN A BATCH SIZE TO COMPLETE ONE EPOCH
	for index in range(0, training_dataset.shape[0], settings.batch_size):
		#set the variable to the last index number of the batch size
		end_batch_size = index + settings.batch_size
		#IF THE END BATCH SIZE EXCEEDS THE TOTAL NUMBER OF DATA, THEN IT IS SET TO NONE
		if end_batch_size >= training_dataset.shape[0] : end_batch_size = None 
		#perform the training session with the specified number of batch
		sess.run(model_vgg.train_step, feed_dict={model_vgg.x : training_dataset[index : end_batch_size], model_vgg.y_ : training_labelset[index : end_batch_size], model_vgg.keep_prob:0.5})
		#sum up all the training accuracy
		training_accuracy += model_vgg.accuracy.eval(session = sess, feed_dict={model_vgg.x : training_dataset[index : end_batch_size], model_vgg.y_ : training_labelset[index : end_batch_size],  model_vgg.keep_prob:1.0})
		#increase the counter
		counter += 1

	# if i % 5 == 0:

		# s = sess.run(model_vgg.merged_summary, feed_dict={model_vgg.x : training_dataset[: settings.batch_size], model_vgg.y_ : training_labelset[: settings.batch_size]})
		

	# writer.add_summary(s, i)

	#GET THE TESTING ACCURACY
	testing_accuracy = testing_session(i, testing_labelset)

	
	#returns the training and testing accuracy in a dictionary format
	return ({
		"training_acc": training_accuracy/counter,
		"testing_acc" : testing_accuracy
		})

#function to perform the testing session
def testing_session(epoch, testing_labelset):

	#variable to keep count how many times a certain size of batch has went into the loop to complete an epoch
	testing_counter = 0
	#to keep the sum of the testing accuracies
	test_accuracy = 0
	#loop through the dataset with the specified step (batch size)
	for index in range(0, X_test.shape[0], settings.accuracy_batch_size):
		#assign the last index of the batch to the variable
		end_batch_size_accuracy = index + settings.accuracy_batch_size
		#if the assigned index number is out of the bound, then it will be set to None 
		if end_batch_size_accuracy >= X_test.shape[0] : end_batch_size_accuracy = None 
		#perform the testing with the specified batch size
		test_accuracy += model_vgg.accuracy.eval(session=sess, feed_dict={model_vgg.x : X_test[index : end_batch_size_accuracy], model_vgg.y_: testing_labelset[index : end_batch_size_accuracy],  model_vgg.keep_prob:1.0})
		#increase the counter
		testing_counter += 1

	# if epoch % 5 == 0:

	# 	s2 = sess.run(model_vgg.merged_summary, feed_dict={model_vgg.X : X_test[: settings.accuracy_batch_size], model_vgg.y_ : one_hot_vector_testingSet[: settings.accuracy_batch_size], model_vgg.keep_prob: 1.0})

	# writer.add_summary(s2, epoch)
	#returns the average testing accuracy
	return(test_accuracy/testing_counter)

#iterate through the defined number of epoch(s)	
for epoch in range(settings.epoch):

	#to keep track of the highest accuracy so far
	highest_accuracy = -1.00

	#these two variables is to keep track of the training and testing accuracy for the defined number of rounds
	#they are then averaged to get the accuracy for 1 full epoch
	epoch_training_acc = 0
	epoch_test_acc = 0

	#iterate through the number of rounds that has been defined "memory_limit"
	for round_number in range(1, settings.memory_limit + 1 ):

		#get the datasets for that particular round [memory_limit] from the function in utils.py file
		X, X_test, Y, Y_test = utils.get_files(round_number)
		#print the current number of round and the epoch
		print("***ROUND*** : ", round_number, " ***EPOCH*** : ", epoch)

		#print the number of training data and testing data for the particular round
		print("Number of training data : ", X.shape)
		print("Number of testing data : ", X_test.shape)

		#initialize the one hot vectors for training and testing set
		one_hot_vector_testingSet = np.zeros((Y_test.shape[0], settings.num_of_classes))
		one_hot_vector_trainingSet = np.zeros((Y.shape[0], settings.num_of_classes))

		#change the labels into one hot vector
		for i in range(Y.shape[0]):

			one_hot_vector_trainingSet[i][int(Y[i])] = 1.0

		for k in range(Y_test.shape[0]):

			one_hot_vector_testingSet[k][int(Y_test[k])] = 1.0


		#get the return values from the function
		accuracies = training_session(X,one_hot_vector_trainingSet, one_hot_vector_testingSet, epoch)

		#sum all the training and testing accuracies together
		epoch_training_acc += float(accuracies['training_acc'])
		epoch_test_acc += float(accuracies['testing_acc'])

	#executed at the end of each epoch
	#print the average of the training and testing accuracy along with the epoch number
	print("EPOCH : " ,epoch, "\nTRAINING ACCURACY : ", epoch_training_acc/settings.memory_limit, "\nTESTING ACCURACY : ", epoch_test_acc/settings.memory_limit)

	#save the model if it produces a higher or equal testing accuracy than the previous epochs
	if epoch_test_acc/settings.memory_limit >= highest_accuracy:
		#save the model at the desired location
		saver.save(sess, save_path)
		#assign the new accuracy to the highest_accuracy variable
		highest_accuracy = epoch_test_acc/settings.memory_limit
		#write the highest accuracy in accuracy_record.txt
		file = open('accuracy_record.txt', 'a')
		file.write("The highest accuracy is : " + str(highest_accuracy) + " \n")
		file.close()


