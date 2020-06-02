# Imports
import numpy as np 
import pickle
import os 

# Hybrid Model Imports
from code.hybrid_model.hybrid_model_utilities import reshape_hybrid_predictions

# GLOBAL VARIABLES
FOLDS = [i for i in range(5)]

# Hybrid Model Results Directory
hybrid_model_results_dir = 'results/hybrid-model/predictions'

# Hybrid Model Raw Predictions Directory
hybrid_model_raw_predictions_dir = os.path.join(hybrid_model_results_dir, 'raw')

# Hybrid Model Reshaped Predictions Directory
hybrid_model_reshape_predictions_dir = os.path.join(hybrid_model_results_dir, 'reshaped')
# Check if it exists or create it
if os.path.isdir(hybrid_model_reshape_predictions_dir) == False:
    os.mkdir(hybrid_model_reshape_predictions_dir)


# Iterate through FOLDS
for fold in FOLDS:
    print("Current fold: {}".format(fold))

    # Open raw predictions filename
    filename = os.path.join(hybrid_model_raw_predictions_dir, 'hybrid_model_preds_CV_Fold_{}.pickle'.format(fold))
    with open(filename, "rb") as fp:
        all_hybrid_predictions = pickle.load(fp)

    # Check shape before the reshape transformation
    print('Shape before reshape transformation: {}'.format(np.shape(all_hybrid_predictions)))

    hybrid_predictions = reshape_hybrid_predictions(hybrid_model_raw_predictions=all_hybrid_predictions)

    # Check shape after the reshape transformations
    print('Shape after reshape transformation: {}'.format(hybrid_predictions.shape))

    # Save new file
    filename = os.path.join(hybrid_model_reshape_predictions_dir, 'hybrid_preds_CV_Fold_{}.pickle'.format(fold))
    with open(filename, "wb") as fp:
        pickle.dump(hybrid_predictions, fp, -1)