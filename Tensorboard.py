# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:39:32 2019

@author: USER
"""
from datetime import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
n_epochs=1000
learning_rate=0.01
#For visualizing
now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,now)

housing = fetch_california_housing()
m,n = housing.data.shape
scaled_housing_data=scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]

X=tf.placeholder(tf.float32,shape=(None,n+1),name="X")
y=tf.placeholder(tf.float32,shape=(None,1),name="y")
theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
y_pred=tf.matmul(X,theta,name="predictions")
error=y_pred-y
mse=tf.reduce_mean(tf.square(error),name="mse")
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(mse)
init=tf.global_variables_initializer()

batch_size=100
n_batches=int(np.ceil(m/batch_size))
mse_summary=tf.summary.scalar('MSE',mse)
file_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())

def fetch_batch(epoch,batch_index,batch_size):
        np.random.seed(epoch * n_batches + batch_index)
                                                          # Set batch_size random indices 
        indices = np.random.randint(m, size=batch_size)  
                                                          # Define a batch X based on previous indices
        X_batch = scaled_housing_data_plus_bias[indices] 
                                                          # y batch 
        y_batch = housing.target.reshape(-1, 1)[indices] #load data from the disc
        
        return X_batch,y_batch
                
    
with tf.Session() as sess:
                sess.run(init)
                
                for epoch in range(n_epochs):
                    for batch_index in range(n_batches):
                        X_batch,y_batch =fetch_batch(epoch,batch_index,batch_size)
                        if batch_index % 10 == 0:
                            summary_str=mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
                            step=epoch*n_batches+batch_index
                            file_writer.add_summary(summary_str,step)
                            
                        sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
                        best_theta=theta.eval()
                        file_writer.close()
                        
