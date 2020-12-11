# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:47:55 2019

@author: USER
"""

import tensorflow as tf
x1=tf.Variable(1)
x1.graph is tf.compat.v1.get_default_graph()

graph=tf.Graph()

with graph.as_default():
    x2=tf.Variable(2)
    