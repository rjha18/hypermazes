import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import hyperfanin_for_kernel, hyperfanin_for_bias, extract_toml, get_log_dir
from tensorflow.keras.regularizers import L2

def setup_model(experiment,batch_size,maps,load=False,eager=False):

    loss_fn = tf.keras.losses.MeanSquaredError()
    toml_data = extract_toml(experiment)
    
    if toml_data['method']=='hyp':
        e_sz = toml_data['e_sz']
        f_sz = toml_data['f_sz']
    else:
        e_sz = None
        f_sz = None
        
    log_dir = get_log_dir(experiment)
    
    model = rlf(batch_size,maps,method=toml_data['method'],e_sz=e_sz,f_sz=f_sz,lr=1e-4)
    callbacks = [keras.callbacks.TensorBoard(log_dir, update_freq=1)]


    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=[loss_fn], run_eagerly=eager
    )
    model.build([(batch_size),(batch_size, 4),(batch_size, 2)])

    if load:
        model.load_weights(log_dir + 'model/weights').expect_partial()


    return model



class rlf(keras.Model):

	def __init__(self,BATCH_SIZE,maps,method='hyp',lr=1e-4,e_sz=None,f_sz=None,classification=False,writer=None):
	
		self.BATCH_SIZE = BATCH_SIZE
		self.classification = classification
		super(rlf, self).__init__()
		self.maps = maps
		self.use_conv = True

		self.decay = L2(1e-5)
		
		self.bottleneck_sz = 128
		
		
		self.writer = writer
		
		self._create_encoder()
		
		if method=='hyp':
			self.hypernet = True
			
			assert f_sz!=None and e_sz!=None
			
			self.e_sz = e_sz
			self.f_sz = f_sz
			#self.e_total = self.total_func_size(2,self.e_sz)
			self.f_total = self.total_func_size(4,self.f_sz)
			
			self._create_hypernet()
		else:
			self.hypernet = False
			self._create_embedding()
			
		
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
		
	def forward_pass(self, inputs, training=False):
		[indices, states, _] = inputs
		
		S = tf.slice(states,[0,0],[-1,2])
		G = tf.slice(states,[0,2],[-1,2])
		
		if self.hypernet:
			theta_f = self.get_theta(indices, training=training)
			
			#e_s = self.func_theta(S,theta_e,self.e_sz,2)
			#e_g = self.func_theta(G,theta_e,self.e_sz,2)
		
			#func_in = tf.concat([e_s,e_g],axis=-1)
			theta = self.func_theta(states,theta_f,self.f_sz,4)
		else:
			z = self.encode(indices, training=training)
			#e_s = self.embed(S,z)
			#e_g = self.embed(G,z)
			
			theta = self.get_angle(S,G,z)
		
		#self.embedding = e_s
		#self.states = S
		
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
			
	
	def _create_encoder(self):
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
		self.norm1 = tf.keras.layers.BatchNormalization()
			
	
	def _create_embedding(self):
	
		#self.emb1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='emb1')
		#self.emb2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='emb2')
		#self.emb3 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='emb3')
		#self.emb4 = tf.keras.layers.Dense(16,activation=None,name='emb4')
		
		self.angle1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='angle1')
		self.angle2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='angle2')
		self.angle3 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='angle3')
		self.angle4 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='angle4')
		self.angle5 = tf.keras.layers.Dense(1,activation=None,name='angle5')
		
	
	def _create_hypernet(self):
		'''
		self.eW = []
		self.eb = []
		fanin = 2
		
		
		for i in range(len(self.e_sz)):
			relu = i<len(self.e_sz)-1
			eWi = tf.keras.layers.Dense(fanin*self.e_sz[i],name='eW'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_kernel(fanin,relu=relu))
			ebi = tf.keras.layers.Dense(1*self.e_sz[i],name='eb'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_bias(relu=relu))
			
			self.eW += [eWi]
			self.eb += [ebi]
			fanin = self.e_sz[i]
		'''
			
			
		self.fW = []
		self.fb = []
		fanin = 4#self.e_sz[-1]
		
		for i in range(len(self.f_sz)):
			relu = i<len(self.e_sz)-1
			fWi = tf.keras.layers.Dense(fanin*self.f_sz[i],name='fW'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_kernel(fanin,relu=relu))
			fbi = tf.keras.layers.Dense(1*self.f_sz[i],name='fb'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_bias(relu=relu))
			
			self.fW += [fWi]
			self.fb += [fbi]
			fanin = self.f_sz[i]
			
		
	def encode(self,indices,training=False):
	
		I = tf.gather(self.maps, tf.cast(indices, tf.int32), axis=0)

		I = tf.transpose(I,[0,2,3,1])
		I = tf.cast(I,tf.float32)
		
		
		if self.use_conv:
			h1 = self.conv1(I)
			h1 += self.conv11(h1) + self.conv12(h1)
			
			h2 = self.conv2(h1)
			h2 += self.conv21(h2) + self.conv22(h2)
			
			h3 = self.conv3(h2)
			h3 += self.conv31(h3) + self.conv32(h3)

			z = self.flat1(h3)
			z = self.fc1(z)
		else:
			z = self.fc3(self.fc2(self.fc1(self.flat1(I))))
		z = self.norm1(z)
		z = self.bottleneck(z)
		self.z = z
		return z
		
	#def embed(self,x,z):
	#	emb_in = tf.concat([x,z],axis=-1)
	#	return self.emb4(self.emb3(self.emb2(self.emb1(emb_in))))
		
	def get_angle(self,s,g,z):
		angle_in = tf.concat([s,g,z],axis=-1)
		return self.angle5(self.angle4(self.angle3(self.angle2(self.angle1(angle_in)))))
		
	def get_theta(self,I,training=False):

		z = self.encode(I,training=training)
		

		'''
		theta_e = []
		
		for i in range(len(self.e_sz)):
			eWi = self.eW[i](z)
			ebi = self.eb[i](z)
			
			theta_e += [tf.concat([eWi,ebi],axis=-1)]
		theta_e = tf.concat(theta_e,axis=-1)
		'''
		
		
		theta_f = []
		
		for i in range(len(self.f_sz)):
			fWi = self.fW[i](z)
			fbi = self.fb[i](z)
			
			theta_f += [tf.concat([fWi,fbi],axis=-1)]
		theta_f = tf.concat(theta_f,axis=-1)
		
		return theta_f


	def get_vars(self):
		var_list = []
	
		for layer in self.layers:
			var_list += layer.trainable_variables;
		return var_list;
		

	def train_step(self, inputs):
		with tf.GradientTape(persistent=False) as tape:
		
			pred = self.forward_pass(inputs, training=True)
			
			loss = self.compiled_loss(inputs[-1], pred)
			
			#loss += 1e-1*self.compiled_loss(self.pts, self.pts_inv)
			
			
			all_vars = self.get_vars()
			
			gradients = tape.gradient(loss, all_vars)
			self.optimizer.apply_gradients(zip(gradients, all_vars))
			# self.compiled_optimizer????

		self.compiled_metrics.update_state(inputs[-1], pred)

		return {m.name: m.result() for m in self.metrics}

	
	
	def test_step(self, inputs):
		pred = self.forward_pass(inputs, training=False)
		self.compiled_metrics.update_state(inputs[-1], pred)

		return {m.name: m.result() for m in self.metrics}

	def call(self, inputs):
		return self.forward_pass(inputs)
		
