# Imports
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np 
import cv2
import time

# Keras Imports
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, multiply, concatenate, Dense, Flatten, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from keras import losses
from keras import backend as K

# ISBI Model Imports
from code.isbi_model.isbi_model_utilities import create_isbi_model, generate_isbi_predictions


# GLOBAL VARIABLES
# FOLDS
FOLDS = [i for i in range(5)]

# ISBI Model Results Directory
isbi_model_results_dir = 'results/isbi-model'

# ISBI Model Weights Directory
isbi_model_weights_dir = os.path.join(isbi_model_results_dir, 'weights')

# ISBI Model Predictions Directory
isbi_model_predictions_dir = os.path.join(isbi_model_results_dir, 'predictions')
if os.path.isdir(isbi_model_predictions_dir) == False:
    os.mkdir(isbi_model_predictions_dir)


# Iterate through folds
for fold in FOLDS:
    print('Current fold: {}'.format(fold))

    # ISBI Model Weights Path
    weights_path = os.path.join(isbi_model_weights_dir, 'isbi_model_trained_Fold_{}.hdf5'.format(fold))

    # Data Paths
    data_path = 'data/resized'
    
    # X_train path
    X_train_path = os.path.join(data_path, 'X_train_221.pickle')

    # X_test path
    X_test_path = os.path.join(data_path, 'X_test_221.pickle')

    # Test indices path
    test_indices_path = 'data/train-test-indices/test_indices_list.pickle'

    
    # Start time measurement for the algorithm performance check
    startTime = time.time()

    # Generate predictions
    isbi_preds = generate_isbi_predictions(
        X_train_path=X_train_path,
        X_test_path=X_test_path,
        test_indices_path=test_indices_path,
        fold=fold,
        isbi_model_weights_path=weights_path
        )
    
    # Time measurement
    print ('The script took {} seconds for fold {}!'.format(time.time() - startTime, fold))

with open(os.path.join(isbi_model_predictions_dir, 'isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold)), 'wb') as f:
    cPickle.dump(isbi_preds, f, -1)

print('ISBI Model Predictions Finished.')