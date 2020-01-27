
import keras
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, AveragePooling2D
from keras.regularizers import l1,l2
from keras.callbacks import CSVLogger,EarlyStopping, Callback
from keras.utils import np_utils, plot_model # utilities for one-hot encoding of ground truth values
from keras import metrics
from keras import backend as K
import numpy as np
import os,sys
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes the tensorflow initial information 

def acc_likelihood(y_true, y_pred):
	return K.mean(K.max(y_pred*y_true,1))

def get_available_gpus():
	local_device_protos = K.get_session().list_devices()
	return np.array([x.name for x in local_device_protos if x.device_type == 'GPU'])


batch_size = 32 # in each iteration, we consider 128 training examples at once
num_epochs = 50 # we iterate twelve times over the entire training set
num_epochs1 = 25 # epochs for the pre-training
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.2 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # there will be 128 neurons in both hidden layers
l1_lambda = 0.0001 # use 0.0001 as a l1-regularisation factor

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)
num_fingers = 10

#(x_train, y_train), (x_test, y_test) = mnist.load_data() # fetch MNIST data
(x_train, y_train), (x_test, y_test) = np.load('mnistdata.npy')

x_train = x_train.reshape(x_train.shape[0], height, width, depth)
x_test = x_test.reshape(x_test.shape[0], height, width, depth)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Normalise data to [0, 1] range
x_test /= 255 # Normalise data to [0, 1] range

c_test = y_test

y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

reps = 11
ssplit = np.array([128,256,512,1024,3200,6400,60000]) # number of examples
oweights = np.array([1,1,1,0.5,0.5,0.5,0.3])
nsplit = ssplit.shape[0]
score = np.zeros(shape=(nsplit,6))
acc1 = np.zeros(shape=(reps,nsplit))
gpus = get_available_gpus().size

#first model - number/finger association
matrix_train = y_train
matrix_test = y_test

for k in range(reps):

	if not os.path.exists('./Logs/'+str(k)):
		os.makedirs('./Logs/'+str(k))

	for i in range(nsplit):
		start = time.time()
		a=k%(x_train.shape[0]/ssplit[i])
		x_split = x_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		y_split = y_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		matrix_split = matrix_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		print('a=',a,'split = ',(ssplit[i]*a),'-',(ssplit[i]*(a+1)),' N = ',x_split.shape[0])
		drop_prob_1 = 0.0
		drop_prob_2 = 0.3

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
		o2 = Dense(num_fingers, activation='softmax', kernel_initializer='glorot_uniform', name="fingers_inout")(o)

		earlyStopping=keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
		model1 = Model(inputs=inp,outputs=o2)

		model1.compile(loss='categorical_crossentropy',optimizer='adam')
		model1.fit(x_split[:ssplit[i],:],matrix_split[:ssplit[i],:],
			epochs=num_epochs1,shuffle=True,callbacks=[earlyStopping],
			verbose=0)

		o = Dense(120, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		o = Dropout(drop_prob_2, name="hidden_dropout1")(o)
		o = Dense(84, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		o = concatenate([o, o2],axis=1,name="concatenate") 
		o = BatchNormalization(name='block_norm2')(o)
		o = Dropout(drop_prob_2, name="hidden_dropout2")(o)
		layerc = Dense(num_classes, kernel_initializer='glorot_uniform', activation='softmax', name='class_output')(o) # Output softmax layer

		model = Model(inputs=[inp],outputs=[layerc,o2])
		#plot_model(model)
		for layer in model.layers:
			if (hasattr(layer, 'rate')):
				layer.rate = layer.rate+0.2 #incresases the dropout rate for the full model

		model.compile(loss={"class_output": 'categorical_crossentropy', "fingers_inout": 'categorical_crossentropy'},
			 		  loss_weights=[1,oweights[i]],
					  optimizer='adam',
					  metrics={"class_output": ['accuracy',acc_likelihood], "fingers_inout": ['mse']})

		csv_logger = CSVLogger('./Logs/'+str(k)+'/training_class2_conv2d'+"{:03d}".format(i)+'.log')
		history = model.fit([x_split], [y_split,matrix_split],
									batch_size=batch_size,
									epochs=num_epochs,
									callbacks=[csv_logger],
									shuffle=True,
									verbose=0,
									validation_data=([x_test], [y_test,matrix_test]))

		score[i] = model.evaluate([x_test], [y_test,matrix_test], verbose=0)
		print('Current split:',i,k)
		print('Test cumulative loss:', score[i][0])
		print('Test classification loss:', score[i][1])
		print('Test finger loss:', score[i][2])
		print('Test classification accuracy:', score[i][3])
		print('Test classification likelihood:', score[i][4])
		print('Test fingers mse:', score[i][5])
		K.clear_session()
		end = time.time()
		print('Elapsed time', end-start)
