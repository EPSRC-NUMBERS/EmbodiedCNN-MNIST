import keras
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D
from keras.regularizers import l1 # l1-regularisation
from keras.callbacks import CSVLogger
from keras.utils import np_utils, plot_model, multi_gpu_model # utilities for one-hot encoding of ground truth values
from keras import metrics
from keras import backend as K
import numpy as np
import os
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or other if you have more;
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes the tensorflow initial information 

def acc_likelihood(y_true, y_pred):
    return K.mean(K.max(y_pred*y_true,1))

def top_2_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2) 
	
def get_available_gpus():
    local_device_protos = K.get_session().list_devices()
    return np.array([x.name for x in local_device_protos if x.device_type == 'GPU'])

batch_size = 128 # in each iteration, we consider 128 training examples at once
num_epochs = 50 # we iterate 20 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.2 # dropout after pooling with probability 0.2
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.25
hidden_size = 128 # there will be 128 neurons in both hidden layers

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 9 classes (1 per digit + zero)

#(x_train, y_train), (x_test, y_test) = mnist.load_data() # fetch MNIST data
(x_train, y_train), (x_test, y_test) = np.load('mnistdata.npy')

x_train = x_train.reshape(x_train.shape[0], height, width, depth)
x_test = x_test.reshape(x_test.shape[0], height, width, depth)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Normalise data to [0, 1] range
x_test /= 255 # Normalise data to [0, 1] range

y_train = np_utils.to_categorical(y_train, num_classes) # Categorical labels - One-hot encode
y_test = np_utils.to_categorical(y_test, num_classes) # same as train

reps=21
ssplit = np.array([128,256,512,1024,3200,6400,60000]) # number of examples
nsplit = ssplit.shape[0]
score = np.zeros(shape=(nsplit,3))

folder = './Logs/'

for k in range(reps):

	if not os.path.exists(folder+str(k)):
		os.makedirs(folder+str(k))

	for i in range(6,nsplit):
		start = time.time()
		#select the subset
		a=k%(x_train.shape[0]/ssplit[i])
		x_split = x_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		y_split = y_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		print('a=',a,'split = ',(ssplit[i]*a),'-',(ssplit[i]*(a+1)),' N = ',x_split.shape[0])

		inp = Input(shape=(height, width, depth)) # N.B. TensorFlow back-end expects channel dimension last
		o = Convolution2D(filters=6, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(inp)
		o = AveragePooling2D(pool_size=(pool_size, pool_size))(o)
		o = BatchNormalization(name='block_normC1')(o)
		o = Dropout(drop_prob_1)(o)
		o = Convolution2D(filters=16, kernel_size=(3, 3),padding='same',  kernel_initializer='he_uniform', activation='relu')(o)
		o = AveragePooling2D()(o)
		o = BatchNormalization(name='block_normC2')(o)
		o = Dropout(drop_prob_1)(o)
		o = Flatten()(o)
		o = Dense(120, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		o = BatchNormalization(name='block_norm1')(o)
		o = Dropout(drop_prob_2)(o)
		o = Dense(84, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		o = BatchNormalization(name='block_norm2')(o)
		o = Dropout(drop_prob_2)(o)
		o = Dense(num_classes, kernel_initializer='glorot_uniform', activation='softmax')(o) # Output softmax layer

		model = Model(inputs=inp, outputs=o) # To define a model, just specify its input and output layers

		plot_model(model)

		model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
		              optimizer='adam', # using the Adam optimiser
		              metrics=['accuracy',top_2_categorical_accuracy,acc_likelihood]) # reporting the accuracy and the likelihood

		csv_logger = CSVLogger(folder+str(k)+'/training_lenet5_'+"{:03d}".format(i)+'.log')

		model.fit(x_split, y_split, # Train the model using the current split of the training set...
		          batch_size=batch_size, epochs=num_epochs,
		          callbacks=[csv_logger],
		          shuffle=True, verbose=0, validation_data=(x_test, y_test)) # Validation data is just for testing each epochs, no stop criterion is applied
		score = model.evaluate(x_test, y_test, verbose=0)
		print('Current split:',i,k)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		print('Test likelihood', score[2])
		K.clear_session()
		end = time.time()
		print('Elapsed time', end-start)