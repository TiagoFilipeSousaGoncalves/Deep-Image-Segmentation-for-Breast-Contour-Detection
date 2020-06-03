# Imports
import numpy as np
import _pickle as cPickle
import os

# Score Utilities Imports
from code.scoring.utilities import scoring, dense_scoring, to_strange_fmt, resize_keypoints_to_original_size


# GLOBAL VARIABLES
FOLDS = [i for i in range(5)]

# Directories
# Data
data_dir = 'data'

# Resized Data
resized_dir = os.path.join(data_dir, 'resized')

# Original Data
original_dir = os.path.join(data_dir, 'original')

# Train and Test Indices
test_indices_list_path = os.path.join(data_dir, 'train-test-indices', 'test_indices_list.pickle')

# ISBI Results Directories
isbi_results_dir = os.path.join('results', 'isbi-model', 'predictions')


# Go trough folds
for fold in FOLDS:
    print('Current fold {}'.format(fold))

    # Open test indices list
    with open(test_indices_list_path, 'rb') as t:
        test_indices_list = cPickle.load(t)

    # Open predictions files
    with open(os.path.join(isbi_results_dir, 'isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold)), 'rb') as f:
        pred_kpts = cPickle.load(f)
    
    # Keypoints were normalized by the image width (512), so we need to "denormalize" them
    pred_kpts = np.array(pred_kpts)
    pred_kpts *= 512


    # print(pred_kpts)

    # Open X data
    # Train
    with open(os.path.join(original_dir, 'X_train_221.pickle'), 'rb') as fp: 
        X_train_original = cPickle.load(fp) 

    # Test
    with open(os.path.join(original_dir, 'X_test_221.pickle'),'rb') as fp: 
        X_test_original = cPickle.load(fp)  

    # Concatenate both to obtain complete dataset
    X_original = np.concatenate((X_train_original, X_test_original))

    # Obtain the images list related to the fold
    X_original = X_original[test_indices_list[fold]]

    # Resized predictions to the original size
    print('Resizing ISBI predictions to original sizes...')
    resized_preds = resize_keypoints_to_original_size(pred_kpts, X_original.copy())
    print('ISBI predictions resized.')


    # Open y data
    # Train
    with open(os.path.join(original_dir, 'y_train_221.pickle'), 'rb') as fp:
        y_train_original = cPickle.load(fp)  

    with open(os.path.join(original_dir, 'y_test_221.pickle'), 'rb') as fp:  
        y_test_original = cPickle.load(fp)  

    # Concatenate both to obtain the complete dataset     
    y_original = np.concatenate((y_train_original, y_test_original))
    
    # Obtain the keypoints related to the fold
    y_original = y_original[test_indices_list[fold]]


    # Create a list append the processing stuff
    print('Creating a list to process...')
    predictions = [] 

    # Go through all the images in this fold
    for i in range(np.shape(X_original)[0]):
        predictions.append(to_strange_fmt(resized_preds[i]))


    # Create a scores list to process
    print('Creating a scores list to process...')
    scores = []

    # Go through all the images in this fold
    for i in range(np.shape(X_original)[0]): 
        scores.append(scoring(predictions=predictions[i], y=y_original[i], img_shape=X_original[i].shape, dataset_diagonal=2701.6085, dataset="221"))
    
    # Generating dense scores
    print('Generating final dense scores...')
    dense_scores = dense_scoring(scores)

    # Print scores
    print('MODEL: {} | FOLD: {}'.format('ISBI', fold))
    print('ENDPOINTS [MEAN, STD, MAX]: {}'.format(dense_scores[0]))
    print('BREAST CONTOUR [MEAN, STD, MAX]: {}'.format(dense_scores[1]))
    print('NIPPLES [MEAN, STD, MAX]: {}\n'.format(dense_scores[2]))

# Finish statement
print("ISBI Model Scoring Results finished.")