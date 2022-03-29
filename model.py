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
    
    if toml_data['method']=='hyp':
        f_sz = toml_data['f_sz']
    else:
        f_sz = None
        
    log_dir = get_log_dir(experiment)
    
    out_num = 8
    
    if toml_data['walls']:
        out_num += 1
    
    model = rlf(batch_size,maps,toml_data['use_conv'],out_num,use_walls=toml_data['walls'],method=toml_data['method'],f_sz=f_sz)
    


    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=loss_fn,
        metrics=[metric_fn], run_eagerly=eager
    )
    model.build([(batch_size),(batch_size, 4),(batch_size,2),(batch_size, out_num),(batch_size,2)])

    if load:
        model.load_weights(log_dir + 'model/weights').expect_partial()


    return model



class rlf(keras.Model):

    def __init__(self,BATCH_SIZE,maps,use_conv,out_num,method='hyp',use_walls=False,lr=1e-4,f_sz=None,classification=False,writer=None):
    
        self.BATCH_SIZE = BATCH_SIZE
        self.classification = classification
        super(rlf, self).__init__()
        self.maps = maps
        self.use_conv = use_conv
        self.use_walls = use_walls
        self.out_num = out_num

        
        self.bottleneck_sz = 128
        
        self.policy_acc = tf.keras.metrics.SparseCategoricalAccuracy('policy_acc')
        
        if self.use_walls:
            self.occupancy_acc = tf.keras.metrics.SparseCategoricalAccuracy('occupancy_acc')
            self._create_cls()
        
        self.writer = writer
        
        self._create_encoder()
        
        if method=='hyp':
            self.hypernet = True
            self.decay_rate = 0.0
            self.decay = L2(self.decay_rate)
            
            assert f_sz!=None
            
            self.f_sz = f_sz
            self.f_total = self.total_func_size(4,self.f_sz)
            
            self._create_hypernet()
        else:
            self.decay_rate = 1e-5
            self.decay = L2(self.decay_rate)
            self.hypernet = False
            self._create_embedding()
            

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
        [indices, states, plain_states, _, _] = inputs
        
        states = tf.cast(states,tf.float32)
        
        S = tf.slice(states,[0,0],[-1,2])
        G = tf.slice(states,[0,2],[-1,2])
        
        
        
        if self.hypernet:
            theta_f = self.get_theta(indices)
            theta = self.func_theta(states,theta_f,self.f_sz,4)
        else:
            z = self.encode(indices)
            theta = self.get_output(S,G,z)
        y = tf.nn.softmax(theta)
        
        if self.use_walls:
            c = tf.nn.softmax(self.cls(self.z,S))
            return y, c    
        else:
            return y    

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
            
    
    def _create_encoder(self):
    
    
        if self.use_conv:
            base_sz = 16
            self.conv0 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=None, padding='same',name='conv0')
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
            base_sz *= 2
            
            self.conv4 = tf.keras.layers.Conv2D(base_sz, [5,5], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same',name='conv4')
            self.conv41 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv41')
            self.conv42 = tf.keras.layers.Conv2D(base_sz, [3,3], strides=(1, 1), activation=tf.nn.leaky_relu, padding='same',name='conv42')        
            
            self.flat1 = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fc1')
        else:
            self.flat1 = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fc1')
            self.fc2 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fc2')
            self.fc3 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='fc3')
        self.bottleneck = tf.keras.layers.Dense(self.bottleneck_sz,activation=None,name='bottleneck')
        self.norm1 = tf.keras.layers.LayerNormalization()

        # idea: topo-normalization. Layer normalization has spherical topology. What about other topologies.

    def _create_embedding(self):
        self.angle1 = tf.keras.layers.Dense(256,activation=tf.nn.leaky_relu,name='angle1')
        self.angle2 = tf.keras.layers.Dense(512,activation=tf.nn.leaky_relu,name='angle2')
        self.angle3 = tf.keras.layers.Dense(512,activation=tf.nn.leaky_relu,name='angle3')
        self.angle4 = tf.keras.layers.Dense(512,activation=tf.nn.leaky_relu,name='angle4')
        self.angle5 = tf.keras.layers.Dense(self.out_num,activation=None,name='angle5')
        
        
    
    def _create_cls(self):
        init = tf.keras.initializers.RandomNormal(stddev=4.0)
        self.B = tf.keras.layers.Dense(64,activation=None,kernel_initializer=init,name='rff')
        self.cls1 = tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu,name='cls1')
        self.cls2 = tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu,name='cls2')
        self.cls3 = tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu,name='cls3')
        self.cls4 = tf.keras.layers.Dense(2,activation=None,name='cls4')
        
    def cls(self, z, y):
        rff = self.B(y)
        rff = tf.reshape(rff,[-1,64,1])
        rff = tf.concat([tf.math.sin(rff),tf.math.cos(rff)],axis=-1)
        rff = tf.reshape(rff,[-1,128])
        cls_in = tf.concat([z,rff],axis=-1)
        return self.cls4(self.cls3(self.cls2(self.cls1(cls_in))  )) 
    
    def _create_hypernet(self):
            
        self.fW = []
        self.fb = []
        fanin = 4
        
        for i in range(len(self.f_sz)):
            relu = i<len(self.f_sz)-1
            fWi = tf.keras.layers.Dense(fanin*self.f_sz[i],name='fW'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_kernel(fanin,relu=relu))
            fbi = tf.keras.layers.Dense(1*self.f_sz[i],name='fb'+str(i+1),kernel_regularizer=self.decay,kernel_initializer=hyperfanin_for_bias(relu=relu))
            
            self.fW += [fWi]
            self.fb += [fbi]
            fanin = self.f_sz[i]
        
    def encode(self,indices):
    
        I = tf.gather(self.maps, tf.cast(indices, tf.int32), axis=0)

        I = tf.transpose(I,[0,2,3,1])
        I = tf.cast(I,tf.float32)
        
        
        if self.use_conv:
            h0 = self.conv0(I)
            
            h1 = self.conv1(h0)
            h1 += self.conv11(h1) + self.conv12(h1)
            
            h2 = self.conv2(h1)
            h2 += self.conv21(h2) + self.conv22(h2)
            
            h3 = self.conv3(h2)
            h3 += self.conv31(h3) + self.conv32(h3)
            
            h4 = self.conv4(h3)
            h4 += self.conv41(h4) + self.conv42(h4)

            z = self.flat1(h4)
            z = self.fc1(z)
        else:
            z = self.flat1(I)
            self.fI = z;
            
            z = self.fc1(z)
            z = self.fc2(z)
            z = self.fc3(z)
            
        z = self.norm1(z)
        z = self.bottleneck(z)
        self.z = z
        return z
        
    def get_output(self,s,g,z):
        net_in = tf.concat([s,g,z],axis=-1)
        
        y = self.angle1(net_in)
        y = self.angle2(y)
        y = self.angle3(y)
        y = self.angle4(y)
        y = self.angle5(y)
        
        return y
        
    def get_theta(self,I):
        z = self.encode(I)
        
        theta_f = []
        
        for i in range(len(self.f_sz)):
            fWi = self.fW[i](z)
            fbi = self.fb[i](z)
            
            theta_f += [tf.concat([fWi,fbi],axis=-1)]
        self.thetas = theta_f
        theta_f = tf.concat(theta_f,axis=-1)
        return theta_f


    def get_vars(self):
        var_list = []
    
        for layer in self.layers:
            var_list += layer.trainable_variables;
        return var_list;
        

    def train_step(self, inputs):
        with tf.GradientTape(persistent=False) as tape:
        
            if self.use_walls:
                yp_hat, yc_hat = self.forward_pass(inputs, training=True)
                policy_loss = self.compiled_loss(inputs[-2], yp_hat)
                occupancy_loss = self.compiled_loss(inputs[-1], yc_hat)
                loss = policy_loss + 1e-3*occupancy_loss
            else:
                yp_hat = self.forward_pass(inputs, training=True)
                loss = self.compiled_loss(inputs[-2], yp_hat)

            if self.hypernet:
                penalty = 0
                for theta in self.thetas:
                    penalty += tf.reduce_mean(tf.reduce_sum(tf.square(theta),axis=-1))
                loss += 1e-5*penalty
            
            
            all_vars = self.get_vars()
            
            gradients = tape.gradient(loss, all_vars)
            self.optimizer.apply_gradients(zip(gradients, all_vars))
            # self.compiled_optimizer????

        
        metrics = {}
        
        self.policy_acc.update_state(inputs[-2], yp_hat)
        metrics[self.policy_acc.name] = self.policy_acc.result()
        
        if self.use_walls:
            self.occupancy_acc.update_state(inputs[-1], yc_hat)
            metrics[self.occupancy_acc.name] = self.occupancy_acc.result()

        return metrics

    
    
    def test_step(self, inputs):
        if self.use_walls:
            yp_hat, yc_hat = self.forward_pass(inputs, training=False)
        else:
            yp_hat = self.forward_pass(inputs, training=False)
            
        metrics = {}
        
        self.policy_acc.update_state(inputs[-2], yp_hat)
        metrics[self.policy_acc.name] = self.policy_acc.result()
        
        if self.use_walls:
            self.occupancy_acc.update_state(inputs[-1], yc_hat)
            metrics[self.occupancy_acc.name] = self.occupancy_acc.result()

        return metrics


    def call(self, inputs):
        return self.forward_pass(inputs)
        
