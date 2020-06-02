# Imports
import numpy as np
import pickle as pkl
import sys
import os
from matplotlib import pyplot as plt
import cv2
import time

# Scipy Imports
import scipy
import scipy.interpolate as interpolate
import scipy.ndimage.morphology as morpho

# Shapely Geometry Imports
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon

# Skimage Imports
import skimage
import skimage.filters as filters

# Sklearn Imports
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Hybrid Model Imports
from code.hybrid_model.hybrid_model_utilities import *


# GLOBAL VARIABLES
# FOLDS
FOLDS = [i for i in range(5)]

# Data Paths
data_path = 'data/resized'

# Train and Test Indices
train_indices_list_path = 'data/train-test-indices/train_indices_list.pickle'
test_indices_list_path = 'data/train-test-indices/test_indices_list.pickle'

# ISBI Model Predictions
isbi_model_predictions_path = 'results/isbi-model/predictions'

# Open y data
y_train_root = os.path.join(data_path, 'y_train_221.pickle')
y_test_root = os.path.join(data_path, 'y_test_221.pickle')

# Open X data
X_train_root = os.path.join(data_path, 'X_train_221.pickle')
X_test_root = os.path.join(data_path, 'X_test_221.pickle')

# Results folder(s)
# Main results directory
results_directory = 'results'
# Check it exists or create it
if os.path.isdir(results_directory) == False:
    os.mkdir(results_directory)


# Model results directory
model_results_directory = os.path.join(results_directory, 'hybrid-model')
# Check if it exists or create it
if os.path.isdir(model_results_directory) == False:
    os.mkdir(model_results_directory)


# Breast contour priors directory
breast_prior_contour_directory = os.path.join(model_results_directory, 'breast-contour-priors')
# Check if it exists or create it
if os.path.isdir(breast_prior_contour_directory) == False:
    os.mkdir(breast_prior_contour_directory)

# Hybrid model predictions directory
hybrid_predictions_directory = os.path.join(model_results_directory, 'predictions')
# Check if it exists or create it
if os.path.isdir(hybrid_predictions_directory) == False:
    os.mkdir(hybrid_predictions_directory)

# Raw predictions directory
raw_predictions_directory = os.path.join(hybrid_predictions_directory, 'raw')
# Check if it exists or create it
if os.path.isdir(raw_predictions_directory) == False:
    os.mkdir(raw_predictions_directory)


# Iterate through FOLDS
for fold in FOLDS:
    print("Current fold: {}".format(fold))
    
    # Generate Breast Contour Priors
    save_breast_contour_params(
        y_train_path=y_train_root,
        y_test_path=y_test_root,
        fold=fold, train_indices_list_path=train_indices_list_path,
        breast_prior_save_path=breast_prior_contour_directory
        )
    print('Breast contour priors saved for fold: {}'.format(fold))

    # Start counting time to check algorithm duration
    startTime = time.time()

    # Open the breast contour priors for this fold
    breast_prior_root = os.path.join(breast_prior_contour_directory, 'breast_contour_prior_CV_Fold_{}.pkl'.format(fold))

    # Open ISBI Model predictions
    with open(os.path.join(isbi_model_predictions_path, 'isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold)), 'rb') as f:
        isbi = pkl.load(f)

    # Convert to Numpy array
    isbi = np.array(isbi)
    # ISBI Model predictions are normalized by width, so we need to "denormalize" them
    isbi *= 512

    # Obtain Hybrid Model Predictions
    generate_hybrid_predictions(
        X_train_path=X_train_root,
        X_test_path=X_test_root,
        test_indices_list_path=test_indices_list_path,
        fold=fold,
        keypoints=isbi,
        breast_prior_path=breast_prior_root,
        result_path=raw_predictions_directory)

    # Compute time per fold
    print('The script took {} seconds for fold {}!'.format(time.time() - startTime, fold))

# Finish statement
print('Hybrid Model Predictions Finished.')