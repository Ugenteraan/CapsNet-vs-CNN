import settings
import tensorflow as tf 


class Model:


	#function to initialize the weight for convolutional layers
	def weight_variable_convolution(self, filter_size, input_depth, output_depth, name):
		#returns a weight variable based on random initialization
		return tf.Variable(tf.truncated_normal([filter_size, filter_size, input_depth, output_depth], stddev = 0.1), name = name)
	#function to initialize the bias variable for the convoltional layers
	def bias_variable_convolution(self, shape, name):
		#returns a randomly initialized bias variable
		return tf.Variable(tf.truncated_normal([shape], stddev = 0.1), name = name)
	#function to initialize the weight variable for the fc layers
	def weight_variable_fc(self, input_size, output_size, name):
		#returns a randomly initialized weight variable
		return tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name = name)
	#function to initialize the bias variable for the fc layers
	def bias_variable_fc(self, shape, name):
		#returns a bias variable initialized with constants
		return tf.Variable(tf.constant(1.0, shape = [shape], name = name))
	#function to create a max_pooling layer
	def max_pooling(self, input_layer, ksize, strides, name):
		#returns a max pool layer with the padding='SAME'
		return tf.nn.max_pool(input_layer, ksize = [1,ksize,ksize,1], strides = [1,strides,strides,1], padding = 'SAME', name = name)
	#function to create a convolution layer with ReLU
	def convolution_layer(self, input_layer, weight, bias, strides, conv_name, bias_name):
		#returns a convolution layer with relu activation
		return tf.nn.relu(tf.nn.conv2d(input_layer, weight, strides = [strides,strides,strides,strides], padding = 'SAME', name = conv_name) + bias, name=bias_name)

	#constructor	
	def __init__(self):
		#set the settings
		image_dimension = settings.picture_input_dimension
		image_depth = settings.image_depth
		conv_weight_filter_size = 9
		patch_depth1, patch_depth2, patch_depth3, patch_depth4 = 64, 128, 256, 512
		num_of_hidden_layer1, num_of_hidden_layer2 = 1152, 512

		#number of pooling layers in the network
		no_of_pooling_layer = 2
		#calculate the dimension of the input size for the fully connected layer after flattening took place
		fc1_input_size = ((image_dimension // (2**no_of_pooling_layer)) * (image_dimension // (2**no_of_pooling_layer))*patch_depth3)

		#initialize the weight and bias variables
		self.wb_variables = {

			#w1-w13 : convolutional layer's weight
			'w1'  : self.weight_variable_convolution(conv_weight_filter_size, image_depth, patch_depth3, 'w1'),

			'w2'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth3, patch_depth3, 'w2'),

			'w3'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth1, patch_depth2, 'w3'),

			'w4'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth2, patch_depth2, 'w4'),

			'w5'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth2, patch_depth3, 'w5'),

			'w6'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth3, patch_depth3, 'w6'),

			'w7'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth3, patch_depth3, 'w7'),

			'w8'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth3, patch_depth4, 'w8'),

			'w9'  : self.weight_variable_convolution(conv_weight_filter_size, patch_depth4, patch_depth4, 'w9'),

			'w10' : self.weight_variable_convolution(conv_weight_filter_size, patch_depth4, patch_depth4, 'w10'),

			'w11' : self.weight_variable_convolution(conv_weight_filter_size, patch_depth4, patch_depth4, 'w11'),

			'w12' : self.weight_variable_convolution(conv_weight_filter_size, patch_depth4, patch_depth4, 'w12'),

			'w13' : self.weight_variable_convolution(conv_weight_filter_size, patch_depth4, patch_depth4, 'w13'),

			#w14-w16 : fully connected layer's weight
			'w14' : self.weight_variable_fc(fc1_input_size, num_of_hidden_layer1, 'w14'),

			'w15' : self.weight_variable_fc(num_of_hidden_layer1, num_of_hidden_layer2, 'w15'),

			'w16' : self.weight_variable_fc(num_of_hidden_layer2, settings.num_of_classes, 'w16'),

			#b1-b13 : convolutional layer's bias
			'b1'  : self.bias_variable_convolution(patch_depth3, 'b1'),

			'b2'  : self.bias_variable_convolution(patch_depth3, 'b2'),

			'b3'  : self.bias_variable_convolution(patch_depth2, 'b3'),

			'b4'  : self.bias_variable_convolution(patch_depth2, 'b4'),

			'b5'  : self.bias_variable_convolution(patch_depth3, 'b5'),

			'b6'  : self.bias_variable_convolution(patch_depth3, 'b6'),

			'b7'  : self.bias_variable_convolution(patch_depth3, 'b7'),

			'b8'  : self.bias_variable_convolution(patch_depth4, 'b8'),

			'b9'  : self.bias_variable_convolution(patch_depth4, 'b9'),

			'b10'  : self.bias_variable_convolution(patch_depth4, 'b10'),

			'b11'  : self.bias_variable_convolution(patch_depth4, 'b11'),

			'b12'  : self.bias_variable_convolution(patch_depth4, 'b12'),

			'b13'  : self.bias_variable_convolution(patch_depth4, 'b13'),

			#b14-b16 : fully connected layer's bias
			'b14'  : self.bias_variable_fc(num_of_hidden_layer1, 'b14'),

			'b15'  : self.bias_variable_fc(num_of_hidden_layer2, 'b15'),

			'b16'  : self.bias_variable_fc(settings.num_of_classes, 'b16')
		}
		#placeholder for the input images
		self.x = tf.placeholder('float', [None, image_dimension, image_dimension])
		#placeholder for the labels
		self.y_ = tf.placeholder('float', [None, settings.num_of_classes])
		#placeholder for the keep probability (to get dropout probability, substract 1 from the probability)
		self.keep_prob = tf.placeholder(tf.float32)
		#reshape the image to be fed into the CNN
		x_image = tf.reshape(self.x, [-1, image_dimension, image_dimension, image_depth])

		#first group of cnn
		with tf.name_scope('Conv-group1'):

			layer1_conv_w_relu = self.convolution_layer(x_image, self.wb_variables['w1'], self.wb_variables['b1'], 1, 'conv1-group1', 'conv1-group1_actv')

			# layer2_conv_w_relu = self.convolution_layer(layer1_conv_w_relu, self.wb_variables['w2'], self.wb_variables['b2'], 1, 'conv2-group1', 'conv2-group1_actv')

		#max_pooling on the output from the last layer of cnn from the first group
		with tf.name_scope('Conv-group1-maxPool'):

			first_pooling_layer = self.max_pooling(layer1_conv_w_relu, 2, 2, 'pooling-1')

		#second group of cnn
		with tf.name_scope('Conv-group2'):

			layer3_conv_w_relu = self.convolution_layer(first_pooling_layer, self.wb_variables['w2'], self.wb_variables['b2'], 1, 'conv1-group2', 'conv1-group2_actv')

			# layer4_conv_w_relu = self.convolution_layer(layer3_conv_w_relu, self.wb_variables['w4'], self.wb_variables['b4'], 1, 'conv2-group2', 'conv2-group2_actv')

		# #max_pooling on the output from the last layer of cnn from the second group
		with tf.name_scope('Conv-group2-maxPool'):

			second_pooling_layer = self.max_pooling(layer3_conv_w_relu, 2, 2, 'pooling-2')

		# #third group of cnn
		# with tf.name_scope('Conv-group3'):

		# 	layer5_conv_w_relu = self.convolution_layer(second_pooling_layer, self.wb_variables['w5'], self.wb_variables['b5'], 1, 'conv1-group3', 'conv1-group3_actv')

		# 	layer6_conv_w_relu = self.convolution_layer(layer5_conv_w_relu, self.wb_variables['w6'], self.wb_variables['b6'], 1, 'conv2-group3', 'conv2-group3_actv')

		# 	layer7_conv_w_relu = self.convolution_layer(layer6_conv_w_relu, self.wb_variables['w7'], self.wb_variables['b7'], 1, 'conv3-group3', 'conv3-group3_actv')

		# #max_pooling on the output from the last layer of cnn from the third group
		# with tf.name_scope('Conv-group3-maxPool'):

		# 	third_pooling_layer = self.max_pooling(layer7_conv_w_relu, 2, 2, 'pooling-3')

		# #fourth group of cnn
		# with tf.name_scope('Conv-group4'):

		# 	layer8_conv_w_relu = self.convolution_layer(third_pooling_layer, self.wb_variables['w8'], self.wb_variables['b8'], 1, 'conv1-group4', 'conv1-group4_actv')

		# 	layer9_conv_w_relu = self.convolution_layer(layer8_conv_w_relu, self.wb_variables['w9'], self.wb_variables['b9'], 1, 'conv2-group4', 'conv2-group4_actv')

		# 	layer10_conv_w_relu = self.convolution_layer(layer9_conv_w_relu, self.wb_variables['w10'], self.wb_variables['b10'], 1, 'conv3-group4', 'conv3-group4_actv')

		# #max pooling on the output from the last layer of cnn from the fourth group
		# with tf.name_scope('Conv-group4-maxPool'):

		# 	fourth_pooling_layer = self.max_pooling(layer10_conv_w_relu, 2, 2, 'pooling-4')

		# #fifth group of cnn
		# with tf.name_scope('Conv-group5'):

		# 	layer11_conv_w_relu = self.convolution_layer(fourth_pooling_layer, self.wb_variables['w11'], self.wb_variables['b11'], 1, 'conv1-group5', 'conv1-group5_actv')

		# 	layer12_conv_w_relu = self.convolution_layer(layer11_conv_w_relu, self.wb_variables['w12'], self.wb_variables['b12'], 1, 'conv2-group5', 'conv2-group5_actv')

		# 	layer13_conv_w_relu = self.convolution_layer(layer12_conv_w_relu, self.wb_variables['w13'], self.wb_variables['b13'], 1, 'conv3-group5', 'conv3-group5_actv')

		# #max pooling on the output from the last layer of cnn from the fifth group
		# with tf.name_scope('Conv-group5-maxPool'):

		# 	fifth_pooling_layer = self.max_pooling(layer13_conv_w_relu, 2, 2, 'pooling-5')

		#flatten the layer
		with tf.name_scope('Flatten-layer'):

			output_shape = (image_dimension // (2**no_of_pooling_layer)) * (image_dimension // (2**no_of_pooling_layer))*patch_depth3	

			self.reshape_output = tf.reshape(second_pooling_layer, [-1, output_shape])

		#fully connected layers
		with tf.name_scope("FC_layers"):

			fc_1_layer = tf.add(tf.matmul(self.reshape_output, self.wb_variables['w14']), self.wb_variables['b14'])

			fc_1_actv = tf.nn.relu(fc_1_layer)

			#dropout in between fc1 and fc2
			with tf.name_scope('Dropout-fc1'):

				dropout_layer1 = tf.nn.dropout(fc_1_actv, self.keep_prob)

			fc_2_layer = tf.add(tf.matmul(dropout_layer1, self.wb_variables['w15']), self.wb_variables['b15'])
			fc_2_actv = tf.nn.relu(fc_2_layer)

			#dropout in between fc2 and output layer
			with tf.name_scope('Dropout-fc2'):

				dropout_layer2 = tf.nn.dropout(fc_2_actv, self.keep_prob)

		#output layer without softmax
		with tf.name_scope('Logits'):

			self.logits = tf.add(tf.matmul(dropout_layer2, self.wb_variables['w16']), self.wb_variables['b16'])

		#output layer with softamx
		with tf.name_scope('Prediction'):

			self.y = tf.nn.softmax(self.logits)

		#perform backpropagation
		with tf.name_scope('Training'):
			#regularization 
			# regularization_value = 0.01 * tf.nn.l2_loss(self.wb_variables['w14']) + 0.01 * tf.nn.l2_loss(self.wb_variables['w15']) + 0.01 * tf.nn.l2_loss(self.wb_variables['w16'])
			#calculate the cost 
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
			#perform training
			self.train_step = tf.train.AdamOptimizer(learning_rate= settings.learning_rate).minimize(self.cost)
		#to calculate accuracy
		with tf.name_scope("Accuracy"):

			correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))

			with tf.name_scope("Accuracy-2"):

				self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))