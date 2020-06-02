# Imports 
import numpy as np 
import _pickle as cPickle
import os
import time

# Segmentation Based Model Imports
from code.segmentation_based_model.keypoints_from_mask_utilities import *

# GLOBAL VARIABLES
# FOLDS
FOLDS = [i for i in range(5)]

# Plot and Save Masks during folds iteration
PLOT_AND_SAVE_MASKS = False

# ISBI Model Directories
isbi_model_results_dir = 'results/isbi-model'

# ISBI Model Predictions Directory
isbi_model_predictions_dir = os.path.join(isbi_model_results_dir, 'predictions')
if os.path.isdir(isbi_model_predictions_dir) == False:
    os.mkdir(isbi_model_predictions_dir)

# Segmentation Based Model Directories
segmentation_based_model_results_dir = 'results/segmentation-based-model'

# Segmentation Based Model U-Net++ Directory
segmentation_based_model_unet_pp_dir = os.path.join(segmentation_based_model_results_dir, 'unet-pp')

# Segmentation Based Model U-Net++ Predicted Masks Directory
segmentation_based_model_unet_pp_masks_dir = os.path.join(segmentation_based_model_unet_pp_dir, 'predictions')

# Segmentation Based Model Final Predictions Directory
segmentation_based_model_predictions = os.path.join(segmentation_based_model_results_dir, 'predictions')
if os.path.isdir(segmentation_based_model_predictions) == False:
    os.mkdir(segmentation_based_model_predictions)


# Iterate through folds
for fold in FOLDS:
    print('Current fold: {}'.format(fold))

    # Open U-Net++ Predicted Masks
    with open(os.path.join(segmentation_based_model_unet_pp_masks_dir, 'unet_pp_preds_CV5_Fold_{}.pickle'.format(fold)), 'rb') as f:
        masks = cPickle.load(f)

    # Open ISBI Model Predicted Keypoints
    with open(os.path.join(isbi_model_predictions_dir, 'isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold)), 'rb') as f:
        isbi = cPickle.load(f)

    # Data preprocessing (masks)
    masks = np.array(masks).reshape((-1, 384, 512))
    masks *= 255

    # If PLOT_AND_SAVE_MASKS == True, this will be activated
    if PLOT_AND_SAVE_MASKS:
        for i in range(masks.shape[0]):
            plt.axis('off')
            plt.imshow(masks[i], cmap='gray')
            plt.savefig('mask'+str(i), bbox_inches='tight', pad_inches=0.0)
            plt.show()


    # Start measuring time
    startTime = time.time()

    # Extract contours from U-Net++ predicted masks
    masks_contours = get_keypoints_from_breasts_masks(masks)
    
    # Uncomment next line if you want to plot masks and detected contours
    # plot_keypoints_from_breast_masks_contours(masks, masks_contours)

    # Generate Segmentation Based Model predictions
    mixed = mix_isbi_and_contours(masks_contours, isbi_preds=isbi)

    # Uncomment next line if you want to plot masks and final segmentation based model predictions
    # plot_keypoints_from_breast_masks_contours(masks, mixed)

    # Convert predictions to ISBI notation
    mixed = mixed_to_our_notation(mixed)

    # Elapsed time computations for algorithm execution performance
    print('The script took {} second for fold {}!'.format(time.time() - startTime, fold))

    # Save predictions file
    with open(os.path.join(segmentation_based_model_predictions, 'mixed_preds_CV_Fold_{}.pickle'.format(fold)), 'wb') as f:
        cPickle.dump(mixed, f, -1)
    
print('Segmentation Based Model Predictions finished.')