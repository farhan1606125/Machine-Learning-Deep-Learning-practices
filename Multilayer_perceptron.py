# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:03:18 2019

@author: USER
"""

import tensorflow as tf
import keras
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
X, y = keras.datasets.mnist.load_data()
    # normalize x



#scaler=StandardScaler()
#X=scaler.fit.X

#mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='files')
#X,y=mnist.data,mnist.target
 # Scale data to [-1, 1] - This is of mayor importance!!!
#X = X/255.0*2 - 1
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf=tf.contrib.learn.DNNClassifier(hidden_units=[300,100],n_classes=10,feature_columns=feature_columns)

dnn_clf.fit(x=X_tain,y=y_train,batch_size=50,steps=40000)

y_pred=list(dnn_clf.predict(X_test))
accuracy_score(y_test,y_pred)

