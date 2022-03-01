import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class rlf(keras.Model):

	def __init__(self,e_sz,f_sz,BATCH_SIZE,maps,lr=1e-4,classification=False,writer=None):
	
		self.BATCH_SIZE = BATCH_SIZE
		self.classification = classification
		super(rlf, self).__init__()
		self.maps = maps
		self.use_conv = False

		self.e_sz = e_sz
		self.f_sz = f_sz
		
		self.bottleneck_sz = 32
		
		self.e_total = self.total_func_size(2,self.e_sz)
		self.f_total = self.total_func_size(2*self.e_sz[-1],self.f_sz)
		
		self.writer = writer
		
		self._create_hypernet()
		
		self.angle_num = 8;

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr);
		if self.classification:
			self.softmax = tf.keras.layers.Softmax()

	def total_func_size(self,in_dim,func_sz):
		total_size = 0

		for i in range(len(func_sz)):
			out_dim = func_sz[i]
			total_size += in_dim * out_dim + out_dim
			in_dim = out_dim
		return total_size
		
		
	def forward_pass(self, inputs):
		[indices, states, _] = inputs
		maps = tf.gather(self.maps, tf.cast(indices, tf.int32), axis=0)

		maps = tf.transpose(maps,[0,2,3,1])
		maps = tf.cast(maps,tf.float32)
		
		S = tf.slice(states,[0,0],[-1,2])
		G = tf.slice(states,[0,2],[-1,2])
		
		theta_e, theta_f = self.get_theta(maps)
		
		e_s = self.func_theta(S,theta_e,self.e_sz,2)
		e_g = self.func_theta(G,theta_e,self.e_sz,2)
		
		self.embedding = e_s
		self.states = S
		
		z = tf.concat([e_s,e_g],axis=-1)
		theta = self.func_theta(z,theta_f,self.f_sz,2*self.e_sz[-1])
		
		x = tf.math.sin(theta)
		y = tf.math.cos(theta)
		
		
		pts = tf.concat([x,y],axis=-1)
		
		return pts	
		

	def func_theta(self,x,theta,sz,in_sz):
	
		num_layers = len(sz)
		
		offset = 0;
		
		y = tf.reshape(x,[-1,1,in_sz])
		
		for i in range(num_layers):
			out_sz = sz[i]
			
			W_sz = in_sz*out_sz
			b_sz = out_sz
			
			W = tf.slice(theta,[0,offset],[-1,W_sz])
			offset += W_sz
			
			b = tf.slice(theta,[0,offset],[-1,b_sz])
			offset += b_sz

			W = tf.reshape(W,[-1,in_sz,out_sz])
			b = tf.reshape(b,[-1,1,out_sz])

			y = tf.matmul(y,W)+b
			
			if i<num_layers-1:
				y = tf.nn.leaky_relu(y)

			in_sz = out_sz

		y = tf.squeeze(y,axis=1)
		if self.classification:
			y = self.softmax(y)

		return y;
			
	
	def _create_hypernet(self):
	
		if self.use_conv:
			base_sz = 16
			self.conv1 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv1')
			self.conv11 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv11')
			self.conv12 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv12')
			
			base_sz *= 2
			self.conv2 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv2')
			self.conv21 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv21')
			self.conv22 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv22')

			base_sz *= 2
			self.conv3 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv3')
			self.conv31 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv31')
			self.conv32 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv32')		
			
			self.flat1 = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,name='fc1')
		else:
			self.flat1 = tf.keras.layers.Flatten()
			self.fc1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fc1')
			self.fc2 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,name='fc2')
			self.fc3 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,name='fc3')
			
		self.bottleneck = tf.keras.layers.Dense(self.bottleneck_sz,activation=None,name='bottleneck')
		self.e_dec = tf.keras.layers.Dense(self.e_total,name='e_dec',kernel_initializer=keras.initializers.RandomNormal(stddev=1e-5))
		self.f_dec = tf.keras.layers.Dense(self.f_total,name='f_dec',kernel_initializer=keras.initializers.RandomNormal(stddev=1e-5))
		
	
	def get_theta(self,I):

		if self.use_conv:
			h1 = self.conv1(I)
			h1 += self.conv11(h1) + self.conv12(h1)
			
			h2 = self.conv2(h1)
			h2 += self.conv21(h2) + self.conv22(h2)
			
			h3 = self.conv3(h2)
			h3 += self.conv31(h3) + self.conv32(h3)

			z = self.flat1(h3)

			z = tf.concat([z,G],axis=-1)
			z = self.fc1(z)
		else:
			z = self.fc3(self.fc2(self.fc1(self.flat1(I))))
		z = self.bottleneck(z)
		
		self.hyperembedding = z

		theta_e = self.e_dec(z)
		theta_f = self.f_dec(z)
		
		return theta_e, theta_f


	def get_vars(self):
		var_list = []
	
		for layer in self.layers:
			var_list += layer.trainable_variables;
		return var_list;
		

	def train_step(self, inputs):
		with tf.GradientTape(persistent=False) as tape:
		
			pred = self.forward_pass(inputs)
			
			loss = self.compiled_loss(inputs[-1], pred)
			
			#loss += 1e-1*self.compiled_loss(self.pts, self.pts_inv)
			
			
			all_vars = self.get_vars()
			
			gradients = tape.gradient(loss, all_vars)
			self.optimizer.apply_gradients(zip(gradients, all_vars))
			# self.compiled_optimizer????

		self.compiled_metrics.update_state(inputs[-1], pred)

		return {m.name: m.result() for m in self.metrics}

	
	
	def test_step(self, inputs):
		pred = self.forward_pass(inputs)
		self.compiled_metrics.update_state(inputs[-1], pred)

		return {m.name: m.result() for m in self.metrics}

	def call(self, inputs):
		return self.forward_pass(inputs)
		
