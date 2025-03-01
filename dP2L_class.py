import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()



#dP2L2 model demo for Journal of Hydrology paper


#This is 
class ScaleLayer_regional_parameterization(Layer):


    def __init__(self, **kwargs):
        super(ScaleLayer_regional_parameterization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.t_mean = self.add_weight(name='t_mean', shape=(1,),  #
                                 initializer=initializers.Constant(value=10.50360728383252),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=10000.0, rate=0.9),
                                 trainable=False)
        self.t_std = self.add_weight(name='t_std', shape=(1,),  #
                                 initializer=initializers.Constant(value=10.30964231561827),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=10000.0, rate=0.9),
                                 trainable=False)

        self.dayl_mean = self.add_weight(name='dayl_mean', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.49992111027762387),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=10000.0, rate=0.9),
                                 trainable=False)
        self.dayl_std = self.add_weight(name='dayl_std', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.08233807739244361),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=10000.0, rate=0.9),
                                 trainable=False)


        super(ScaleLayer_regional_parameterization, self).build(input_shape)

    def call(self, inputs):
        #met(气象输入) = [wrap_number_train, wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        print("ScaleLayer_regional_parameterization_Inputs_Shape_PET",inputs.shape)

        met = inputs[:,:,:2]

        self.t_scaled = (met[:,:,0:1] - self.t_mean) / self.t_std
        self.dayl_scaled = (met[:,:,1:2] - self.dayl_mean) / self.dayl_std


        self.met_scaled = K.concatenate((self.t_scaled, self.dayl_scaled), axis=-1)

        attrs = inputs[:,:,2:]

        return  K.concatenate((self.met_scaled, attrs), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
class LSTM_parameterization(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM_parameterization, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM_parameterization, self).build(input_shape)


    def reset_parameters(self):
        #self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        #self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        #self.bias.initializer = initializers.Constant(value=0)
        #self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]
        #print('forcing_shape:',forcing.shape)
        #attrs = inputs_x[1]     #[batch, dim]
        #print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('forcing_seqfir_shape:',forcing_seqfir.shape)

        #attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('attrs_seqfir_shape:',attrs_seqfir.shape)


        seq_len = forcing_seqfir.shape[0]
        #print('seq_len:',seq_len)
        batch_size = forcing_seqfir.shape[1]
        #print('batch_size:',batch_size)

        #init_states = [K.zeros((K.shape(forcing)[0], 2))]
        #h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        #print("bias_batch:",bias_batch.shape)

        #bias_s_batch = K.expand_dims(self.bias_s, axis=0)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=batch_size, axis=0)
        #这里对静态变量通过输入门的相加操作可能有问题,两张量维度不一样, attrs输入这里应该是二维 [batch_size, xs_dim]   , [sample_size, xs_dim]
        #i = K.sigmoid(K.dot(attrs, self.w_sh) + bias_s_batch)

        for t in range(seq_len):
            h_0, c_0 = h_x

            #这里也有问题, 必须把forcing数据的seq_len放在第一维 [seq_len, batch_size, xd_dim]
            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)

            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        return h_n, c_n
class regional_dP2L(Layer):

    def __init__(self, mode='normal', h_nodes=64, seed=200, **kwargs):
        self.mode = mode
        self.h_nodes = h_nodes
        self.seed = seed
        super(regional_dP2L, self).__init__(**kwargs)

    def build(self, input_shape):



        self.prnn_w1 = self.add_weight(name='prnn_w1',
                                       shape=(256, self.h_nodes),
                                       initializer=initializers.RandomUniform(seed=self.seed - 5),

                                       trainable=True)

        self.prnn_b1 = self.add_weight(name='prnn_b1',
                                       shape=(self.h_nodes,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w2 = self.add_weight(name='prnn_w2',
                                       shape=(self.h_nodes, 16),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b2 = self.add_weight(name='prnn_b2',
                                       shape=(16,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.prnn_w3 = self.add_weight(name='prnn_w3',
                                       shape=(16, 1),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.prnn_b3 = self.add_weight(name='prnn_b3',
                                       shape=(1,),
                                       initializer=initializers.zeros(),
                                       trainable=True)






        self.para_w1 = self.add_weight(name='para_w1',
                                       shape=(27, 128),
                                       initializer=initializers.RandomUniform(seed=self.seed - 5),

                                       trainable=True)

        self.para_b1 = self.add_weight(name='para_b1',
                                       shape=(128,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.para_w2 = self.add_weight(name='para_w2',
                                       shape=(128, self.h_nodes),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.para_b2 = self.add_weight(name='para_b2',
                                       shape=(self.h_nodes,),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.para_w3 = self.add_weight(name='para_w3',
                                       shape=(self.h_nodes, 16),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.para_b3 = self.add_weight(name='para_b3',
                                       shape=(16,),
                                       initializer=initializers.zeros(),
                                       trainable=True)
        self.para_w4 = self.add_weight(name='para_w4',
                                       shape=(16, 6),
                                       initializer=initializers.RandomUniform(seed=self.seed + 5),

                                       trainable=True)

        self.para_b4 = self.add_weight(name='para_b4',
                                       shape=(16,),
                                       initializer=initializers.zeros(),
                                       trainable=True)


        self.shape = input_shape

        super(regional_dP2L, self).build(input_shape)

    def heaviside(self, x):


        return (K.tanh(5 * x) + 1) / 2




    def rainsnowpartition(self, p, t, tmin):

        tmin = tmin * -3  # scale (0, 1) into (-3, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]


    def snowbucket(self, s0, t, ddf, tmax):

        ddf = ddf * 5            # scale (0, 1) into (0, 5)
        tmax = tmax  * 3          # scale (0, 1) into (0, 3)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt


    def soilbucket(self, s1, pet, f, smax, qmax):

        f = f / 10                 # scale (0, 1) into (0, 0.1)
        smax = smax * 1400 + 100   # scale (0, 1) into (100, 1500)
        qmax = qmax * 40 + 10      # scale (0, 1) into (10, 50)
        pet = pet * 29.9 + 0.1          # scale (0, 1) into (0.1, 30.0)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)

        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]  # Snowpack
        s1 = states[0][:, 1:2]  # Soilwater

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        # Load the current paras
        tmin = step_in[:, 3:4]
        tmax = step_in[:, 4:5]
        ddf  = step_in[:, 5:6]
        f    = step_in[:, 6:7]
        smax = step_in[:, 7:8]
        qmax = step_in[:, 8:9]




        [_ps, _pr] = self.rainsnowpartition(p, t, tmin)

        _m = self.snowbucket(s0, t, ddf, tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, f, smax, qmax)


        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]


    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]  # daily precipitation
        tmean = inputs[:, :, 1:2] # daily mean temperature
        dayl = inputs[:, :, 2:3]  # daily Daylength



        attrs = inputs[:,:,5:]  # 27 dimensions

        # Calculate PET using Hamon’s formulation - EXPHYDRO
        #pets = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)


        # Learning PET using differentiable learning (LSTM-NN)
        pet_inputs = K.concatenate((tmean, dayl, attrs), axis=-1)
        pet_inputs_scale = ScaleLayer_regional_parameterization(name='ScaleLayer_regional_parameterization_PET')(pet_inputs)
        pet_hn, pet_cn = LSTM_parameterization(input_xd=29, hidden_size=256, seed=200)(pet_inputs_scale)
        pet0 = K.tanh(K.dot(pet_hn, self.prnn_w1)+ self.prnn_b1) # layer 1
        pet1 = K.tanh(K.dot(pet0, self.prnn_w2)+ self.prnn_b2) # layer 2
        pets = K.sigmoid(K.dot(pet1, self.prnn_w3)+ self.prnn_b3) # layer 3
        pets = K.permute_dimensions(pets, pattern=(1, 0, 2))


        # Learning hydrological parameters using differentiable learning (NN)
        parameters = K.tanh(K.dot(attrs, self.para_w1)+ self.para_b1) # layer 1
        parameters = K.tanh(K.dot(parameters, self.para_w2)+ self.para_b2) # layer 2
        parameters = K.tanh(K.dot(parameters, self.para_w3)+ self.para_b3) # layer 2
        parameters = K.sigmoid(K.dot(parameters, self.para_w4)+ self.para_b4) # layer 3


        # Concatenate prcp, tmean, and pets into a new input
        new_inputs = K.concatenate((prcp, tmean, pets, parameters), axis=-1)
        print("new_inputs", new_inputs.shape)


        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        # return 3 outputs:
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        tmin = parameters[:, :, 0:1]
        tmax = parameters[:, :, 1:2]
        ddf  = parameters[:, :, 2:3]
        f    = parameters[:, :, 3:4]
        smax = parameters[:, :, 4:5]
        qmax = parameters[:, :, 5:6]


        # Calculate final process variables
        [psnow, prain] = self.rainsnowpartition(prcp, tmean, tmin)

        m = self.snowbucket(s0, tmean, ddf, tmax)

        [et, qsub, qsurf] = self.soilbucket(s1, pets, f, smax, qmax)

        effective_rainfall = m + prain


        if self.mode == "normal":
            return K.concatenate([effective_rainfall, s1, et], axis=-1)
        elif self.mode == "analysis":
            return K.concatenate([s0, m, et, s1], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 3)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 4)
class LSTMq(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTMq, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTMq, self).build(input_shape)


    def reset_parameters(self):
        #self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        #self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        #self.bias.initializer = initializers.Constant(value=0)
        #self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]
        #print('forcing_shape:',forcing.shape)
        #attrs = inputs_x[1]     #[batch, dim]
        #print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('forcing_seqfir_shape:',forcing_seqfir.shape)

        #attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('attrs_seqfir_shape:',attrs_seqfir.shape)


        seq_len = forcing_seqfir.shape[0]
        #print('seq_len:',seq_len)
        batch_size = forcing_seqfir.shape[1]
        #print('batch_size:',batch_size)

        #init_states = [K.zeros((K.shape(forcing)[0], 2))]
        #h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        #print("bias_batch:",bias_batch.shape)

        #bias_s_batch = K.expand_dims(self.bias_s, axis=0)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=batch_size, axis=0)
        #这里对静态变量通过输入门的相加操作可能有问题,两张量维度不一样, attrs输入这里应该是二维 [batch_size, xs_dim]   , [sample_size, xs_dim]
        #i = K.sigmoid(K.dot(attrs, self.w_sh) + bias_s_batch)

        for t in range(seq_len):
            h_0, c_0 = h_x

            #这里也有问题, 必须把forcing数据的seq_len放在第一维 [seq_len, batch_size, xd_dim]
            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)

            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        return h_n
