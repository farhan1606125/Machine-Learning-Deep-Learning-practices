# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:12:40 2019

@author: USER
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

#load sample images

dataset=np.array(load_sample_images().images,dtype=np.float32)
batch_size,height,width,channels = dataset.shape

#create 2 filters 
filters_test = np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters_test[:,3,:,0] = 1 #vertical_line
filters_test[:,3,:,1] = 1#Horizontal_line
#create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.nn.conv2d(X,filters_test,strides=[1,2,2,1],padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution,feed_dict={X: dataset})
plt.imshow(dataset[0,:,:,0]) 
plt.show() 
print("TADA")   
plt.imshow(dataset[0,:,:,2])
plt.show() 
print("TADA") 
plt.imshow(output[0,:,:,0])
plt.show() 
print("TADA") 
plt.imshow(output[0,:,:,0])
print("TADA") 
plt.show()    
