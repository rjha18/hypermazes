import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import hyperfanin_for_kernel, hyperfanin_for_bias, extract_toml, get_log_dir
from tensorflow.keras.regularizers import L2








def setup_model(experiment,batch_size,maps,load=False,eager=False):

    toml_data = extract_toml(experiment)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metric_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    
    log_dir = get_log_dir(experiment)
    
    out_num = 8
    
    model = rlf(batch_size,maps,toml_data['prim_sz'],bottleneck=toml_data['bottleneck'],hypernet=toml_data['hypernet'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=[metric_fn], run_eagerly=eager
    )
    model.build([(batch_size),(batch_size, 4),(batch_size, out_num)])

    if load:
        model.load_weights(log_dir + 'model/weights').expect_partial()


    return model



class rlf(keras.Model):

    def __init__(self,BATCH_SIZE,maps,prim_sz,bottleneck=128,hypernet=False,lr=1e-4,classification=False,writer=None):
    
        self.BATCH_SIZE = BATCH_SIZE
        self.classification = classification
        super(rlf, self).__init__()
        self.maps = maps

        self.H = 31
        self.W = 31

        self.out_num = 8

        self.prim_sz = prim_sz
        self.hypernet = hypernet
        self.bottleneck = bottleneck
        
        self.decay_rate = 1e-5
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr);
        
        if self.hypernet:
            self.decay = L2(0.0)
            self._create_hypernet()
        else:
            self.decay = L2(self.decay_rate)
            Ng = self.total_func_size(4,prim_sz)
            ratio = (128+Ng)/(128+prim_sz[0])
            self.embedding_dim =  np.int32(np.ceil(bottleneck*ratio))
            self._create_embedding()
            
            
        self._create_encoder()

        self.policy_acc = tf.keras.metrics.SparseCategoricalAccuracy('policy_acc')
        self.writer = writer

        
        

        
    def forward_pass(self, inputs, training=False):
        [indices, states, _] = inputs
        
        I = tf.gather(self.maps, tf.cast(indices, tf.int32), axis=0)

        I = tf.transpose(I,[0,2,3,1])
        I = tf.slice(I,[0,0,0,0],[-1,-1,-1,1])
        I = tf.cast(I,tf.float32)
        
        
        states = tf.cast(states,tf.float32)
        S = tf.slice(states,[0,0],[-1,2])
        G = tf.slice(states,[0,2],[-1,2])
        
        z_img = self.encoder(I)
        if self.hypernet:
            theta,self.theta_weights = self.hyp_model(z_img)
            
            y = self.prim_model([S,G,theta])
        else:
            y = self.prim_model([S,G,z_img])
            
        return y    






    def train_step(self, inputs):
        with tf.GradientTape(persistent=False) as tape:
            
            y_hat = self.forward_pass(inputs, training=True)
            loss = self.compiled_loss(inputs[2], y_hat)

            if self.hypernet:
                penalty = tf.reduce_mean(tf.reduce_sum(tf.square(self.theta_weights),axis=-1))
                loss += self.decay_rate*penalty
                
                all_vars = self.encoder.trainable_variables + self.hyp_model.trainable_variables
                grad = tape.gradient(loss, all_vars)
                self.optimizer.apply_gradients(zip(grad,all_vars))
            
            else:
                all_vars = self.encoder.trainable_variables + self.prim_model.trainable_variables
                grad = tape.gradient(loss, all_vars)
                self.optimizer.apply_gradients(zip(grad,all_vars))

        
        metrics = {}
        self.policy_acc.update_state(inputs[2], y_hat)
        metrics[self.policy_acc.name] = self.policy_acc.result()
        
        return metrics

    
    
    def test_step(self, inputs):
    
        y_hat = self.forward_pass(inputs, training=False)
            
        metrics = {}
        
        self.policy_acc.update_state(inputs[2], y_hat)
        metrics[self.policy_acc.name] = self.policy_acc.result()
        
        return metrics


    def call(self, inputs):
        return self.forward_pass(inputs)
        
        
        
        
        
    def _create_embedding(self):
        S = tf.keras.layers.Input(shape=(2,))
        G = tf.keras.layers.Input(shape=(2,))
        z_img = tf.keras.layers.Input(shape=(self.embedding_dim,))
    
        y = tf.concat([S,G,z_img],axis=-1)
        for i in range(len(self.prim_sz)-1):
            y = tf.keras.layers.Dense(self.prim_sz[i],activation=tf.nn.leaky_relu,kernel_regularizer=self.decay,name='prim'+str(i))(y)
        y = tf.keras.layers.Dense(self.out_num,activation=None,kernel_regularizer=self.decay,name='prim_logits')(y)
        y = tf.nn.softmax(y)
        
        self.prim_model = tf.keras.Model([S,G,z_img],y)
        
        
    def _create_hypernet(self):
        self.prim_weights = []
        self.prim_biases = []
        
        fanin = 4
        
        for i in range(len(self.prim_sz)):
            relu = i<(len(self.prim_sz)-1)
            self.prim_weights += [tf.keras.layers.Dense(fanin*self.prim_sz[i],name='fW'+str(i+1),kernel_initializer=hyperfanin_for_kernel(fanin,relu=relu))]
            self.prim_biases += [tf.keras.layers.Dense(1*self.prim_sz[i],name='fb'+str(i+1),kernel_initializer=hyperfanin_for_bias(relu=relu))]
            fanin = self.prim_sz[i]
        
        z_img = tf.keras.layers.Input(shape=(self.bottleneck,))
        
        theta_prim = []
        theta_weight = []
        
        for i in range(len(self.prim_sz)):
            Wi = self.prim_weights[i](z_img)
            bi = self.prim_biases[i](z_img)
            theta_weight += [Wi]
            theta_prim += [Wi]
            theta_prim += [bi]
            
            
        theta_weight = tf.concat(theta_weight,axis=-1)
        theta_prim = tf.concat(theta_prim,axis=-1)
        
        self.hyp_model = tf.keras.Model(z_img,[theta_prim,theta_weight])
        
        self.prim_num = self.total_func_size(4,self.prim_sz)
        
        S = tf.keras.layers.Input(shape=(2,))
        G = tf.keras.layers.Input(shape=(2,))
        theta = tf.keras.layers.Input(shape=(self.prim_num,))
        
        y = self.func_theta(tf.concat([S,G],axis=-1),theta,self.prim_sz,4)
        y = tf.nn.softmax(y)
        
        self.prim_model = tf.keras.Model([S,G,theta],y)
        
        


    
    def _create_encoder(self):
    
        base_sz = 8
        conv0 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=None, padding='same',name='conv0')
        conv1 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv1')
        conv11 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv11')
        conv12 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv12')
        
        base_sz *= 2
        conv2 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv2')
        conv21 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv21')
        conv22 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv22')

        base_sz *= 2
        conv3 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv3')
        conv31 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv31')
        conv32 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv32')    
        base_sz *= 2
        
        conv4 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv4')
        conv41 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv41')
        conv42 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv42')        
        
        flat1 = tf.keras.layers.Flatten()
        fc1 = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu,name='fc1')
        norm1 = tf.keras.layers.LayerNormalization()
        
        if self.hypernet:
            fc2 = tf.keras.layers.Dense(self.bottleneck,activation=tf.nn.leaky_relu,name='fc2')
        else:
            fc2 = tf.keras.layers.Dense(self.embedding_dim,activation=tf.nn.leaky_relu,name='fc2')
            
        norm2 = tf.keras.layers.LayerNormalization()


        I = tf.keras.layers.Input(shape=(self.H,self.W,1,))
    
        h0 = conv0(I)
        
        h1 = conv1(h0)
        h1 += conv11(h1) + conv12(h1)
        
        h2 = conv2(h1)
        h2 += conv21(h2) + conv22(h2)
        
        h3 = conv3(h2)
        h3 += conv31(h3) + conv32(h3)
        
        h4 = conv4(h3)
        h4 += conv41(h4) + conv42(h4)

        z = flat1(h4)
        z = fc1(z)
        z = norm1(z)
        z = fc2(z)
        z = norm2(z)
        
        self.encoder = tf.keras.Model(I,z)







    def total_func_size(self,in_dim,func_sz):
        total_size = 0

        for i in range(len(func_sz)):
            out_dim = func_sz[i]
            total_size += in_dim * out_dim + out_dim
            in_dim = out_dim
        return total_size
        
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

        return y;
        
        
