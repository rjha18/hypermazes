

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd



class rlf(keras.Model):

	def __init__(self,e_sz,f_sz,fnm):
		super(rlf, self).__init__()
		
		try:
			fp = open(fnm, 'r')
			fp.close()
		except OSError:
			print("Map file cannot be opened.")
			raise OSError()
			
		self.map_data = np.array(pd.read_csv(fnm,header=None,delimiter=' '));
		self.map_sz = np.prod(self.map_data.shape)
		
		self.e_sz = e_sz
		self.f_sz = f_sz
		
		self.bottleneck_sz = 64
		
		self.e_total = self.total_func_size(2,self.e_sz)
		self.f_total = self.total_func_size(2+2,self.f_sz)
		
		self.writer = tf.summary.create_file_writer('./logs/{}'.format(OUTDIR))
		
		self._create_hypernet()
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3);
		
		
		states = np.zeros((0,2))
		
		for ih in range(self.map_data.shape[0]):
			for iw in range(self.map_data.shape[1]):
				cell = self.map_data[ih,iw]
				
				if cell==0:
					state = np.array([ih,iw]).reshape([1,2])
					states = np.concatenate([states,state],axis=0)
		self.states = states;
		self.angle_num = 8;
		

	def total_func_size(self,in_dim,func_sz):
		total_size = 0

		for i in range(len(func_sz)):
			out_dim = func_size[i]
			total_size += in_dim * out_dim + out_dim
			in_dim = out_dim
		return total_size
		

	
		
	def _MSE(self,y,y_hat):
		N = self.cfg.BATCH_SIZE
		y_flat = tf.reshape(y,[N,-1])
		y_hat_flat = tf.reshape(y_hat,[N,-1])
		return tf.reduce_sum(tf.square(y_flat-y_hat_flat),axis=-1)



		
	def forward_pass(self, inputs):
		[I,angles] = inputs;
		
		
		theta_e, theta_f = self.get_theta(I)


		e_s = self.func_theta(self.states,theta_e,self.e_sz,2)
		print(e_s)
		input()

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
		
		return y;
			
			
	
	
	
	def _create_hypernet(self):
		self.flatten = tf.keras.layers.Flatten(name='flatten')
		self.enc1 = tf.keras.layers.Dense(256,activation=tf.nn.elu,name='enc1')
		self.enc2 = tf.keras.layers.Dense(256,activation=tf.nn.elu,name='enc2')
		self.enc3 = tf.keras.layers.Dense(256,activation=tf.nn.elu,name='enc3')
		self.bottleneck = tf.keras.layers.Dense(self.bottleneck_sz,activation=tf.nn.elu,name='bottleneck')
		self.e_dec = tf.keras.layers.Dense(self.e_total,name='e_dec')
		self.f_dec = tf.keras.layers.Dense(self.f_total,name='f_dec')
		
	
	def get_theta(self,I):
		I_flat = self.flatten(I)
		z = self.enc1(I_flat)
		z = self.enc2(z)
		z = self.enc3(z)
		z = self.bottleneck(z)
		
		theta_e = self.e_dec(z)
		theta_f = self.f_dec(z)
		
		return theta_e, theta_f
	
	

	def train_step(self, inputs):
	
		with tf.GradientTape(persistent=False) as tape:

			self.forward_pass(inputs)
				
			'''
			fp_loss = tf.reduce_mean(self.pred_error());
			#fa_loss,baseline_loss,fa_loss_no_baseline = self.REINFORCE();
			fa_loss = self.REINFORCE();
			
			_,cls_loss = self.compute_cls_loss()
			cls_loss = tf.reduce_mean(cls_loss)
				
				
				
				
			#baseline_vars = self.get_baseline_vars()
			cls_vars = self.get_cls_vars()
			state_vars = self.get_state_vars()
			action_vars = self.get_action_vars()
			

			### Sanity check
			all_vars = cls_vars+state_vars+action_vars;
			
			total_size = 0.0
			
			for vr in all_vars:
				vr_sz = np.prod(vr.get_shape().as_list())
				total_size += vr_sz
			#print(total_size)
			#input()
			
			loss = fa_loss + fp_loss + cls_loss
				
				
			gradients = tape.gradient(loss, all_vars)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, 1.0)
			self.optimizer1.apply_gradients(zip(gradients, all_vars))
			'''

		
		with self.writer.as_default():

			tf.summary.scalar('fp_loss',fp_loss,self.optimizer1.iterations);
			#tf.summary.scalar('fa_loss',fa_loss_no_baseline,self.optimizer1.iterations);
			#tf.summary.scalar('baseline_loss',baseline_loss,self.optimizer1.iterations);
			tf.summary.scalar('cls_loss',cls_loss,self.optimizer1.iterations);
			tf.summary.scalar('loss',loss,self.optimizer1.iterations);


		pred = self.activations['y_pred'][-1]
		self.compiled_metrics.update_state(tf.tile(inputs[1],[self.cfg.M,]), pred)

		return {m.name: m.result() for m in self.metrics}

	
	
	def test_step(self, inputs):
		self.forward_pass(inputs, False)
		pred = self.activations['y_pred'][-1]
		self.compiled_metrics.update_state(tf.tile(inputs[1],[self.cfg.M,]), pred)

		return {m.name: m.result() for m in self.metrics}


	def call(self, inputs, training=False):
		return self.forward_pass(inputs, training)
		
	
		
			
