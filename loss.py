
import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):

    y_pred = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)


    #Add a small enough number to the denominator to prevent Nan Loss
    return numerator / (denominator+0.01)


def nse_metrics(y_true, y_pred):


    y_pred = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)

    #Add a small enough number to the denominator to prevent Nan Loss
    loss =  numerator / (denominator+0.01)

    return 1.0 - loss
