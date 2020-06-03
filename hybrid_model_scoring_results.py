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

# Hybrid Results Directories
hybrid_results_dir = os.path.join('results', 'hybrid-model', 'predictions', 'reshaped')


# Go trough folds
for fold in FOLDS:
    print('Current fold {}'.format(fold))

    # Open test indices list
    with open(test_indices_list_path, 'rb') as t:
        test_indices_list = cPickle.load(t)

    # Open predictions files
    # ISBI Predictions
    with open(os.path.join(isbi_results_dir, 'isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold)), 'rb') as f:
        isbi_pred_kpts = cPickle.load(f)
    
    # Keypoints were normalized by the image width (512), so we need to "denormalize" them
    isbi_pred_kpts = np.array(isbi_pred_kpts)
    isbi_pred_kpts *= 512


    # Hybrid Predictions
    with open(os.path.join(hybrid_results_dir, 'hybrid_preds_CV_Fold_{}.pickle'.format(fold)), 'rb') as f:
        hybrid_pred_kpts = cPickle.load(f)

    # print(hybrid_pred_kpts)

    # Convert to NumPy array
    hybrid_pred_kpts = np.array(hybrid_pred_kpts)
    # print(hybrid_predictions.shape)
    
    # We have to be sure that the Hybrid Predictions of Endpoints and Nipples are the same as ISBI's
    # Endpoints
    # truth_lp = y[0:2]
    # truth_midl = y[32:34]
    # truth_midr = y[66:68]
    # truth_rp = y[34:36]
    
    # Nipples
    # truth_l_nipple = y[70:72]
    # truth_r_nipple = y[72:74]
    
    # Begin Process Statement
    print('Updating Hybrid Model Endpoints for scoring purposes...')
    
    # Iterate through ISBI predictions and update Hybrid Predictions Accordingly
    for index, value in enumerate(isbi_pred_kpts):
        # Endpoints
        hybrid_pred_kpts[index][0:2] = value.copy()[0:2]
        # hybrid_predictions[index][0:2] = np.round(hybrid_predictions[index][0:2], 5)
        # hybrid_predictions[index][1] = value[1]
        
        hybrid_pred_kpts[index][32:34] = value.copy()[32:34]
        # hybrid_predictions[index][32:34] = np.round(hybrid_predictions[index][32:34], 5)
        # hybrid_predictions[index][33] = value[33]
        
        # Mid-points are inversed in hybrid predictions
        hybrid_pred_kpts[index][66:68] = value.copy()[66:68]
        # hybrid_predictions[index][66:68] = np.round(hybrid_predictions[index][66:68], 5)
        # hybrid_predictions[index][67] = value[35]
        
        hybrid_pred_kpts[index][34:36] = value.copy()[34:36]
        # hybrid_predictions[index][34:36] = np.round(hybrid_predictions[index][34:36], 5)
        # hybrid_predictions[index][35] = value[67]
        
        # Nipples
        hybrid_pred_kpts[index][70:72] = value.copy()[70:72]
        # hybrid_predictions[index][71] = value[71]
        hybrid_pred_kpts[index][72:74] = value.copy()[72:74]
        # hybrid_predictions[index][73] = value[73] 
    
    print('Hybrid Model Endpoints updated.')


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
    print('Resizing Hybrid predictions to original sizes...')
    resized_preds = resize_keypoints_to_original_size(hybrid_pred_kpts, X_original.copy())
    print('Hybrid predictions resized.')


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
    print('MODEL: {} | FOLD: {}'.format('Hybrid', fold))
    print('ENDPOINTS [MEAN, STD, MAX]: {}'.format(dense_scores[0]))
    print('BREAST CONTOUR [MEAN, STD, MAX]: {}'.format(dense_scores[1]))
    print('NIPPLES [MEAN, STD, MAX]: {}\n'.format(dense_scores[2]))

# Finish statement
print("Hybrid Model Scoring Results finished.")