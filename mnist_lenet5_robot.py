
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


def acc_likelihood(y_true, y_pred):
	return K.mean(K.max(y_pred*y_true,1))

def get_available_gpus():
	local_device_protos = K.get_session().list_devices()
	return np.array([x.name for x in local_device_protos if x.device_type == 'GPU'])

def top_2_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2) 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes the tensorflow initial information 


batch_size = 32 # in each iteration, we consider 128 training examples at once
num_epochs = 50 # we iterate twelve times over the entire training set
num_epochs1 = 25 # epochs for the pre-training
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.2 # dropout after pooling with probability 0.2
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # there will be 128 neurons in both hidden layers
l1_lambda = 0.0001 # use 0.0001 as a l1-regularisation factor

num_train = 60000 # there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
num_classes = 10 # there are 10 classes (1 per digit)
num_fingers = 16

#(x_train, y_train), (x_test, y_test) = mnist.load_data() # fetch MNIST data
(x_train, y_train), (x_test, y_test) = np.load('mnistdata.npy')

x_train = x_train.reshape(x_train.shape[0], height, width, depth)
x_test = x_test.reshape(x_test.shape[0], height, width, depth)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # Normalise data to [0, 1] range
x_test /= 255 # Normalise data to [0, 1] range

matrix_train = np.load('train_robot'+".npy")
matrix_test = np.load('test_robot'+".npy")

y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

reps = 21
ssplit = np.array([128,256,512,1024,3200,6400,60000]) # number of examples
oweights = np.array([1,1,0.8,0.4,0.3,0.3,0.1])
nsplit = ssplit.shape[0]
score = np.zeros(shape=(nsplit,7))
acc1 = np.zeros(shape=(reps,nsplit))
gpus = get_available_gpus().size

#first model - number/finger association
inp2 = Input(shape=(num_fingers,))
out2 = Dense(num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax', name='class_output2')(inp2) # Output softmax layer
model2 = Model(inputs=inp2,outputs=out2)	
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.fit(matrix_train,y_train,epochs=6,shuffle=True,verbose=0)
out_weights2 = model2.get_layer('class_output2').get_weights()

folder = './Logs/'

for k in range(reps):

	if not os.path.exists(folder+str(k)):
		os.makedirs(folder+str(k))

	for i in range(nsplit):
		start = time.time()
		a=k%(x_train.shape[0]/ssplit[i])
		x_split = x_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		y_split = y_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
		matrix_split = matrix_train[(ssplit[i]*a):(ssplit[i]*(a+1))]
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
		o2 = Dense(num_fingers, activation='sigmoid', kernel_initializer='random_normal', name="fingers_inout")(o)
			#kernel_regularizer=l1(l1_lambda), #bias_regularizer=l1(l1_lambda),bias_initializer='zeros',

		o = Dense(120, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		o = BatchNormalization(name='block_norm1')(o)
		o = Dropout(drop_prob_2, name="hidden_dropout1")(o)
		o = Dense(84, kernel_initializer='he_uniform', activation='relu')(o) # Hidden ReLU layer
		# o = BatchNormalization(name='block_norm2')(o)
		# o = Dropout(drop_prob_2, name="hidden_dropout2")(o)
		o = concatenate([o, o2],axis=1,name="concatenate") 
		layerc = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform', name='class_output')(o) # Output softmax layer

		model = Model(inputs=[inp],outputs=[layerc,o2])

		model.compile(loss={"class_output": 'categorical_crossentropy', "fingers_inout": 'binary_crossentropy'},
			 		  loss_weights=[1,oweights[i]],
					  optimizer='adam',
					  metrics={"class_output": ['accuracy',top_2_categorical_accuracy,acc_likelihood], "fingers_inout": ['mse']})

		hidden_size=84
		out_weights = model.get_layer('class_output').get_weights()
		#out_weights[0][:hidden_size,:] = 0
		out_weights[0][hidden_size:,:] = out_weights2[0]
		out_weights[1] = out_weights2[1]
		model.get_layer('class_output').set_weights(out_weights)

		batch_size = 32 
		if (i>5):
			batch_size = 128 

		csv_logger = CSVLogger(folder+str(k)+'/training_robotP_conv2d'+"{:03d}".format(i)+'.log')
		history = model.fit([x_split], [y_split,matrix_split],
									batch_size=batch_size,
									epochs=num_epochs,
									callbacks=[csv_logger],
									shuffle=True,
									verbose=0,
									validation_data=([x_test], [y_test,matrix_test]))

		print('Current split:',i,k)
		print('Test classification accuracy:', history.history['val_class_output_acc'][0])
		print('Test classification likelihood:', history.history['val_class_output_acc_likelihood'][0])
		print('Test classification top2 acc:', history.history['val_class_output_top_2_categorical_accuracy'][0])
		print('-------')
		print('Test classification accuracy:', history.history['val_class_output_acc'][num_epochs-1])
		print('Test classification likelihood:', history.history['val_class_output_acc_likelihood'][num_epochs-1])
		print('Test classification top2 acc:', history.history['val_class_output_top_2_categorical_accuracy'][num_epochs-1])
		K.clear_session()
		end = time.time()
		print('Elapsed time', end-start)
