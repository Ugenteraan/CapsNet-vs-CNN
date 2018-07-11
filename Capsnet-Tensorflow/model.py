import tensorflow as tf 
import numpy as np 
import settings as cfg



class Model:

	def squash(self, X):
		vec_squared_norm = tf.reduce_sum(tf.square(X), -2, keep_dims=True)
		scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
		return scalar_factor * X

	def routing(self, X, b_IJ):

		batch_size = cfg.batch_size

		W = tf.Variable(tf.truncated_normal([1, self.X_shape_at_digitcaps, self.num_of_cls, 8, 16], stddev=1e-1))
		X = tf.tile(X, [1, 1, self.num_of_cls, 1 ,1])
		W = tf.tile(W, [batch_size, 1, 1, 1, 1])

		u_hat = tf.matmul(W, X, transpose_a=True)
		u_hat_stopped = tf.stop_gradient(u_hat)

		for i in range(self.routing_iter):

			c_IJ = tf.nn.softmax(b_IJ, dim=2)

			if i == self.routing_iter - 1:
				s_J = tf.multiply(c_IJ, u_hat)
				s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
				v_J = self.squash(s_J)
			else:
				s_J = tf.multiply(c_IJ, u_hat_stopped)
				s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
				v_J = self.squash(s_J)
				v_J_tiled = tf.tile(v_J, [1, self.X_shape_at_digitcaps, 1, 1, 1])
				u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
				b_IJ += u_produce_v

		return v_J

	def primaryCaps(self, X, num_output, num_vector, kernel=None, stride=None):

		batch_size = cfg.batch_size

		capsules = tf.contrib.layers.conv2d(X, num_output * num_vector, kernel, stride, padding='VALID', activation_fn=tf.nn.relu)
		capsules = tf.reshape(capsules, (batch_size, -1, num_vector, 1))

		return self.squash(capsules)

	def digitCaps(self, X, num_output):

		batch_size = cfg.batch_size

		X_ = tf.reshape(X, shape=(batch_size, -1, 1, X.shape[-2].value, 1))
		b_IJ = tf.constant(np.zeros([self.batch_size, self.X_shape_at_digitcaps, num_output, 1, 1], dtype=np.float32))
		capsules = self.routing(X_, b_IJ)
		capsules = tf.squeeze(capsules, axis=1)
		return capsules

	def __init__(self):

		self.X_shape_at_digitcaps = cfg.X_shape_digitCaps
		self.batch_size = cfg.batch_size
		self.reg_scale = cfg.regularization_scale
		self.epsilon = cfg.epsilon
		self.learning_rate = cfg.learning_rate
		self.m_minus = cfg.m_minus
		self.m_plus = cfg.m_plus
		self.lambda_val = cfg.lambda_val
		self.inp_img_dim = cfg.picture_input_dimension
		self.inp_img_depth = cfg.image_depth
		self.num_of_cls = cfg.num_of_classes
		self.routing_iter = cfg.routing_iter

		self.x = tf.placeholder(tf.float32, shape=(None, self.inp_img_dim, self.inp_img_dim))
		self.y_= tf.placeholder(tf.float32, shape=(None, self.num_of_cls))

		self.keep_prob = tf.placeholder(tf.float32)

		self.x_reshape = tf.reshape(self.x, [-1,50,50,1])

		conv1 = tf.contrib.layers.conv2d(self.x_reshape, num_outputs=256, kernel_size=9, stride=1, padding='VALID', activation_fn=tf.nn.relu)

		caps1 = self.primaryCaps(conv1, num_output=32, num_vector=8, kernel=9, stride=2)

		caps2 = self.digitCaps(caps1, self.num_of_cls)

		v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + self.epsilon)

		self.logits = tf.nn.softmax(v_length, dim=1)[:,:,0,0]

		max_l = tf.square(tf.maximum(0., self.m_plus - v_length))

		max_r = tf.square(tf.maximum(0., v_length - self.m_minus))

		max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
		max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

		L_c = self.y_ * max_l + self.lambda_val * (1 - self.y_) * max_r
		margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
		origin = tf.reshape(self.x, shape=(self.batch_size, -1))
		self.cost = margin_loss 
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
