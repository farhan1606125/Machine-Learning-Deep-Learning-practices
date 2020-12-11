
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
n_epochs=1000
scaler=StandardScaler()
 
housing=fetch_california_housing()
m,n=housing.data.shape
housing_data_plus_bias=np.c_[np.ones((m,1)),housing.data]
scaled_housing_data_plus_bias=housing_data_plus_bias
learning_rate=0.01
X=tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name="X")
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")
theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
y_pred = tf.matmul(X,theta,name="predictions")
error=y_pred-y
mse=tf.reduce_mean(tf.square(error),name="mse")
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)           #optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9) Momentum Optimizer                                                 #gradients=2/m*tf.matmul(tf.transpose(X),error)
training_op=optimizer.minimize(mse)                                               #tf.assign(theta,theta - learning_rate*gradients)
 
init=tf.global_variables_initializer()
 
with tf.Session() as sess:
     sess.run(init)
     
     for epoch in range(n_epochs):
         if epoch % 100 == 0:
             print("Epoch",epoch,"MSE =",mse.eval())
         sess.run(training_op)
     best_theta=theta.eval()
     
     
     
#Using autodiff
gradients=tf.gradients(mse,[theta])[0]     
print(mse)