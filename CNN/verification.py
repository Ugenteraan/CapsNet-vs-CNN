import tensorflow as tf 
import numpy as np 
import model
from settings import *
import cv2

tf.reset_default_graph()

sess = tf.InteractiveSession()

model_vgg = model.Model()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

try:

	saver.restore(sess, os.getcwd() + '/ckpt_folder/model.ckpt')

except:

	print("Model has not been loaded!")


image = cv2.imread('test.jpg')

resized = cv2.resize(image, (picture_input_dimension, picture_input_dimension))

input_image = np.resize(resized, (1, picture_input_dimension, picture_input_dimension, image_depth))

result = sess.run(model_vgg.y, feed_dict={model_vgg.x: input_image, model_vgg.keep_prob:1.0})

theresult = folders[np.argmax(result)]

print(theresult)

sess.close()