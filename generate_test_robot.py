from __future__ import print_function

import keras
from keras.datasets import mnist
import numpy as np

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

c_train = y_train
c_test = y_test

robot_inputs =np.random.random_sample((10,16))
robot_inputs[1:10,:] = np.genfromtxt('robot.cvs',delimiter=',').transpose()
num_fingers = robot_inputs.shape[1]

y_robot = keras.utils.to_categorical(np.array(range(num_fingers)))

s=0
matrix_train = np.zeros((c_train.size,num_fingers))
for x in c_train[:]:
	matrix_train[s]= robot_inputs[x]
	s=s+1

s=0
matrix_test = np.zeros((c_test.size,num_fingers))
for x in c_test[:]:
	matrix_test[s]=robot_inputs[x]
	s=s+1

np.save('train_robot',matrix_train)

np.save('test_robot',matrix_test)