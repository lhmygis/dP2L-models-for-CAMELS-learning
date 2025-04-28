import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf

## Import libraries developed by this study
from dP2L_class import regional_dP2L
from dataprocess import DataforIndividual
import loss

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = "../the project path"  #The root directory where all .py files are located
attrs_path = "../the project path/CAMELS_attributes/attributedata_671.csv" #csv file of standardized values of 27 basin attributes for 531 CAMELS basins


# Training Period
training_start = '1990-10-01'
training_end = '2010-09-30'


# CAMELS basin ID used for training. 
#If you want to train all basins, add all CAMELS basin IDs to the basin_id array. 
#In this example, we only provide basin streamflow data for HUC=01 and 02. For other HUC data, please download the complete CAMELS dataset and add other HUC data to the path:"../the project path/camels_data/usgs_streamflow/.."
basin_id = [
'1022500',
'1031500',
'1047000',
'1052500',
'1054200',
'1055000',
'1057000',
'1073000',
'1078000',
'1123000',
'1134500',
    ...
    ...
]


all_list = []
#all_list1 = []

for i in range(len(basin_id)):
    a = basin_id[i]

    if len(basin_id[i]) == 7:
        basin_id[i] = '0' + basin_id[i]

    hydrodata = DataforIndividual(working_path, basin_id[i]).load_data()

    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    #train_set1 = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]

    if a.startswith('0'):
        single_basin_id = a[1:]

    else:
        single_basin_id = a
        
    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('gauge_id')
    rows_bool = (static_x.index == int(single_basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)
    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)
    result = np.concatenate((train_set, local_static_x_for_train), axis=-1)
    result1 = np.concatenate((train_set1, local_static_x_for_train), axis=-1)
    all_list.append(result)
    all_list1.append(result1)

result_ = all_list[0]
#result1_ = all_list1[0]

for i in range(len(all_list)-1):
    result_ = np.concatenate((result_, all_list[i+1]), axis=0)
    #result1_ = np.concatenate((result1_, all_list1[i+1]), axis=0)

print(result_.shape)
#print(result1_.shape)

sum_result = result_[:,
             [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 5]]


def generate_train_test(train_set, wrap_length):
    train_set_ = pd.DataFrame(train_set)
    train_x_np = train_set_.values[:, :-1]


    print("prcp_mean:", np.mean(train_x_np[:, 0:1]))
    print("tmean_mean:", np.mean(train_x_np[:, 1:2]))
    print("dayl_mean:", np.mean(train_x_np[:, 2:3]))
    print("srad_mean:", np.mean(train_x_np[:, 3:4]))
    print("vp_mean:", np.mean(train_x_np[:, 4:5]))

    print("prcp_std:", np.std(train_x_np[:, 0:1]))
    print("tmean_std:", np.std(train_x_np[:, 1:2]))
    print("dayl_std:", np.std(train_x_np[:, 2:3]))
    print("srad_std:", np.std(train_x_np[:, 3:4]))
    print("vp_std:", np.std(train_x_np[:, 4:5]))


    train_y_np = train_set_.values[:, -1:]

    print("Q_mean:",  np.mean(train_y_np[:,-1:]))
    print("Q_std:",  np.std(train_y_np[:,-1:]))


    wrap_number_train = (train_x_np.shape[0] - wrap_length) // 1 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))


    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 1:(wrap_length + i * 1), :]
        train_y[i, :, :] = train_y_np[i * 1:(wrap_length + i * 1), :]

    return train_x, train_y


wrap_length = 270  # It can be other values, but recommend this value should not be less than 180 days
train_x, train_y = generate_train_test(sum_result, wrap_length=wrap_length)

print(f'The shape of train_x, train_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_y.shape}')



def create_model(input_xd_shape, hodes, seed):
    #Setting batch_size=1 within xd_input_for_dP2L means that during each training epoch, the model is updated one sample at a time. 
    #You can also set it to other values, but the batch size should evenly divide the total number of samples.
    
    xd_input_for_dP2L = Input(shape=input_xd_shape, batch_size=1, name='Input_xd1') 
    hydro_output = regional_dP2L(mode='normal', h_nodes = hodes, seed = seed, name='Regional_dP2L_PUB')(xd_input_for_dP2L)
    
    model = Model(inputs=xd_input_for_dP2L, outputs=hydro_output)
    return model


def train_model(model, train_x1, train_y, ep_number, lrate, save_path):
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()

    model.compile(loss= loss.nse_loss, metrics=[loss.nse_metrics],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lrate,clipnorm=1.0))

    history = model.fit(x=train_x1, y=train_y, epochs=ep_number, batch_size=1,
                        callbacks=[save, es, reduce, tnan])
    return history



#Model storage path
save_path_models = f"../the project path/Models_h5/dP2L.h5"


model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]),
                     hodes = 64, seed = 101)
model.summary()

prnn_ealstm_history = train_model(model=model, train_x1=train_x,
                                  train_y=train_y, ep_number=100, lrate=0.001, save_path=save_path_models)
