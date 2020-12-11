# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 02:12:33 2019

@author: USER
"""

import tensorflow as tf
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(b,c)

f = tf.subtract(d,e)

sess=tf.Session()
outs=sess.run(f)
sess.close()
print("outs={}".format(outs))

