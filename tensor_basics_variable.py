# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 02:30:29 2019

@author: USER
"""

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)
x = tf.placeholder(tf.float32)

linear_model=w*x+b

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

print(sess.run(linear_model,{x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)

squared_deltas=tf.square(linear_model-y)

loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[-1,-2,-3,-4]})

print(sess.run([w,b]))    

