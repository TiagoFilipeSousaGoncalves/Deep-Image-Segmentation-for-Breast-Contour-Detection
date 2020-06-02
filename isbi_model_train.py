# Imports
import _pickle as cPickle
import numpy as np 
import os 
import cv2

# Keras Imports
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, multiply, concatenate, Dense, Flatten, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from keras import losses
from keras import backend as K

# ISBI Model Imports
from code.isbi_model.isbi_model_train_generator import Generator
from code.isbi_model.isbi_model_utilities import create_isbi_model

# CUDA Environment Variables (adapt them to your personal settings)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# GLOBAL VARIABLES
# ISBI Model Results Directory
# Main Results Directory
results_directory = 'results'
if os.path.isdir(results_directory) == False:
        os.mkdir(results_directory)

# ISBI Model Results Directory 
isbi_model_results_dir = os.path.join(results_directory, 'isbi-model')
if os.path.isdir(isbi_model_results_dir) == False:
        os.mkdir(isbi_model_results_dir)

# ISBI Model Weights Directory
isbi_model_results_weights_dir = os.path.join(isbi_model_results_dir, 'weights')
if os.path.isdir(isbi_model_results_weights_dir) == False:
        os.mkdir(isbi_model_results_weights_dir)

# Data directory
data_dir = 'data/resized'

# X data (images)
with open(os.path.join(data_dir, 'X_train_221.pickle'), 'rb') as fp:
        X_train = cPickle.load(fp)

with open(os.path.join(data_dir, 'X_test_221.pickle'), 'rb') as fp: 
        X_test = cPickle.load(fp)

X = np.concatenate((X_train, X_test))

# Heatmaps data (heatmaps)
with open(os.path.join(data_dir, 'heatmaps_train_221.pickle'), 'rb') as fp: 
        heatmaps_train = cPickle.load(fp)

with open(os.path.join(data_dir, 'heatmaps_test_221.pickle'), 'rb') as fp: 
        heatmaps_test = cPickle.load(fp)

heatmaps = np.concatenate((heatmaps_train, heatmaps_test))

# y data (keypoints)
with open(os.path.join(data_dir, 'y_train_221.pickle'), 'rb') as fp: 
        y_train = cPickle.load(fp)

with open(os.path.join(data_dir, 'y_test_221.pickle'), 'rb') as fp: 
        y_test = cPickle.load(fp)

y = np.concatenate((y_train, y_test))


# Open train indices for the CV-5 fold training
with open('data/train-test-indices/train_indices_list.pickle', 'rb') as fp:
        train_indices_list = cPickle.load(fp)


# Data preprocessing
# Prepare the image for the VGG model
X = np.array(X, dtype='float')
X = preprocess_input(X)
# print(np.max(X),'maximum of X')
# print(np.min(X),'minimum of X')

# Reshape heatmaps array for the U-Net model
heatmaps = np.array(heatmaps)
heatmaps = heatmaps.reshape((-1, 384, 512, 1))

# Normalized keypoints data by the image width (512)
y = np.array(y)
y /= 512

# Flip-indices variable to the Generator Class (data augmentation by horizontal flips)
flip_indices = [(0,34),(1,35),(2,36),(3,37),(4,38),(5,39),(6,40),(7,41),\
(8,42),(9,43),(10,44),(11,45),(12,46),(13,47),(14,48),(15,49),(16,50),(17,51),\
(18,52),(19,53),(20,54),(21,55),(22,56),(23,57),(24,58),(25,59),(26,60),(27,61),(28,62),(29,63),(30,64),(31,65),\
(32,66),(33,67),(68,68),(69,69),(70,72),(71,73)]


# Iterate through train indices list
for index in range(np.shape(train_indices_list)[0]):
        print('Current fold: {}'.format(index))

        # Create ISBI Model object
        model = create_isbi_model()

        # Create Generator object
        my_generator = Generator(X[train_indices_list[index]],
                                heatmaps[train_indices_list[index]],
                                y[train_indices_list[index]],
                                batchsize=2,
                                flip_ratio=0.5,
                                translation_ratio=0.5,
                                rotate_ratio = 0.5,
                                flip_indices=flip_indices
                                )


        model.fit_generator(my_generator.generate(), steps_per_epoch = my_generator.size_train/2, epochs=300, verbose=2)
        model.save_weights(os.path.join(isbi_model_results_weights_dir, 'isbi_model_trained_Fold_{}.hdf5'.format(index)))