# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:09:47 2019

@author: USER
"""

with tf.name.scope("loss") as scope :
    error=y_pred-y
    mse=tf.reduce_mean(tf.square(error),name="mse")