import tensorflow as tf
import numpy as np
import random
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
import matplotlib.pyplot as plt
import seaborn
from PIL import Image
import glob
import os

latent_dim = 10
inputs_decoder = 24

X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

class dataset:

	def __init__(self, ls):

		self.lenght = len(ls)
		self.dataset = ls
		self.index = -1

	def next_batch(self, batch_size):

		self.index += 1

		if self.index * batch_size > self.lenght:

			random.shuffle(self.dataset)
			self.index = 0

		return self.dataset[self.index*batch_size:(self.index+1)*batch_size]		

def leaky_relu(x, alpha=0.3):
	return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X, keep_prob, reuse=None):

	with tf.variable_scope("encoder"):

		X = tf.reshape(X, shape=[-1, 28, 28, 1])
		conv1 = tf.layers.conv2d(X, filters=16, kernel_size=7, strides=1, padding='same', activation=leaky_relu)
		drop1 = tf.nn.dropout(conv1, keep_prob)

		conv2 = tf.layers.conv2d(drop1, filters=32, kernel_size=5, strides=2, padding='same', activation=leaky_relu)
		drop2 = tf.nn.dropout(conv2, keep_prob)

		conv3 = tf.layers.conv2d(drop2, filters=64, kernel_size=3, strides=2, padding='same', activation=leaky_relu)
		drop3 = tf.nn.dropout(conv3, keep_prob)

		conv4 = tf.layers.conv2d(drop3, filters=128, kernel_size=3, strides=1, padding='same', activation=leaky_relu)
		drop4 = tf.nn.dropout(conv4, keep_prob)
		flat = tf.contrib.layers.flatten(drop4)
	
		mean = tf.layers.dense(flat, units=latent_dim)
		std = 0.5 * tf.layers.dense(flat, units=latent_dim)
		epsilon = tf.random_normal([tf.shape(flat)[0], latent_dim])
		z = mean + tf.multiply(epsilon, tf.exp(std))

	return z, mean, std

def decoder(sampled_z, keep_prob, reuse=None):

	with tf.variable_scope("decoder"):

		layer1 = tf.layers.dense(sampled_z, units=inputs_decoder, activation=leaky_relu)
		layer2 = tf.layers.dense(layer1, units=inputs_decoder * 2 + 1, activation=leaky_relu)
		layer2_reshaped = tf.reshape(layer2, [-1, 7, 7, 1])

		conv1 = tf.layers.conv2d_transpose(layer2_reshaped, filters=128, kernel_size=3, strides=1, padding='same', activation=leaky_relu)
		drop1 = tf.nn.dropout(conv1, keep_prob)

		conv2 = tf.layers.conv2d_transpose(drop1, filters=64, kernel_size=3, strides=2, padding='same', activation=leaky_relu)
		drop2 = tf.nn.dropout(conv2, keep_prob)

		conv3 = tf.layers.conv2d_transpose(drop2, filters=32, kernel_size=5, strides=2, padding='same', activation=leaky_relu)
		drop3 = tf.nn.dropout(conv3, keep_prob)

		conv4 = tf.layers.conv2d_transpose(conv3, filters=16, kernel_size=7, strides=1, padding='same', activation=leaky_relu)
		drop4 = tf.nn.dropout(conv4, keep_prob)

		flat = tf.contrib.layers.flatten(drop4)

		img_flat = tf.layers.dense(flat, units=28*28, activation=tf.nn.sigmoid)
		img = tf.reshape(img_flat, shape=[-1, 28, 28])

	return img