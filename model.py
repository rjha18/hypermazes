

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd



class rlf(keras.Model):

	def __init__(self,e_sz,f_sz,fnm,BATCH_SIZE,lr=1e-4,classification=False,writer=None):
	
		self.BATCH_SIZE = BATCH_SIZE
		self.classification = classification
		super(rlf, self).__init__()
		
		try:
			fp = open(fnm, 'r')
			fp.close()
		except OSError:
			print("Map file cannot be opened.")
			raise OSError()
			
		self.map_data = np.array(pd.read_csv(fnm,header=None,delimiter=' '));
		self.M = self.map_data.shape[0]
		self.map_sz = np.prod(self.map_data.shape)
		
		self.e_sz = e_sz
		self.f_sz = f_sz
		
		self.bottleneck_sz = 32
		
		self.e_total = self.total_func_size(2,self.e_sz)
		self.f_total = self.total_func_size(self.e_sz[-1],self.f_sz)
		
		self.writer = writer
		
		self._create_hypernet()
		
		
		
		states = np.zeros((0,2))
		
		for ih in range(self.map_data.shape[0]):
			for iw in range(self.map_data.shape[1]):
				cell = self.map_data[ih,iw]
				
				if cell==0:
					state = np.array([ih,iw]).reshape([1,2])
					states = np.concatenate([states,state],axis=0)
		self.states = states.astype(np.float32);
		self.angle_num = 8;
		
		self.I = tf.constant(self.map_data)

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
		[maps, states, _] = inputs
		theta_e, theta_f = self.get_theta(maps)
		
		S = tf.slice(states,[0,0],[-1,2])
		G = tf.slice(states,[0,2],[-1,2])
		
		e_s = self.func_theta(S,theta_e,self.e_sz,2)
		e_g = self.func_theta(G,theta_e,self.e_sz,2)
		
		self.embedding = e_s
		self.states = S
		
		z = e_g-e_s#tf.concat([e_s,e_g],axis=-1)
		res = self.func_theta(z,theta_f,self.f_sz,self.e_sz[-1])
		
		'''
		
		
		norm_s = tf.sqrt(tf.reduce_sum(tf.square(e_s),axis=-1,keepdims=True))+1e-4
		norm_g = tf.sqrt(tf.reduce_sum(tf.square(e_g),axis=-1,keepdims=True))+1e-4
		dot = tf.reduce_sum(e_s*e_g,axis=-1,keepdims=True)
		
		res = tf.math.acos(dot/(norm_s*norm_g))
		
		'''
		
		x = tf.slice(res,[0,0],[-1,1])
		y = tf.slice(res,[0,1],[-1,1])
		
		x = tf.math.cos(x)
		y = tf.math.sin(y)
		
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
				y = tf.nn.elu(y)

			in_sz = out_sz

		y = tf.squeeze(y,axis=1)
		if self.classification:
			y = self.softmax(y)

		return y;
			
	
	def _create_hypernet(self):
		self.enc1 = tf.keras.layers.Dense(128,activation=tf.nn.elu,name='enc1')
		self.enc2 = tf.keras.layers.Dense(128,activation=tf.nn.elu,name='enc2')
		self.enc3 = tf.keras.layers.Dense(128,activation=tf.nn.elu,name='enc3')
		self.bottleneck = tf.keras.layers.Dense(self.bottleneck_sz,name='bottleneck')
		self.e_dec = tf.keras.layers.Dense(self.e_total,name='e_dec')
		self.f_dec = tf.keras.layers.Dense(self.f_total,name='f_dec')
		
	
	def get_theta(self,I):
	
		N = tf.shape(I)[0]
		I_flat = tf.reshape(I,[N,-1])

		z = self.enc1(I_flat)
		z = self.enc2(z)
		z = self.enc3(z)
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
		
