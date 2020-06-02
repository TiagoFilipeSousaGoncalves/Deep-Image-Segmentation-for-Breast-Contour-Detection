# Imports
import numpy as np 
import _pickle as cPickle
import os 
import time 

# Tensorflow Imports
import tensorflow as tf

# Tensorflow trick to solve some compatibility issues 
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Keras Imports
from keras import backend as K

# Segmentation Based Model Imports
from code.segmentation_based_model.unet_plus_plus_model.segmentation_models import Xnet

# CUDA Environment Variables (adapt them to your personal settings)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# GLOBAL VARIABLES
# FOLDS
FOLDS = [i for i in range(5)]

# Data Directory
data_directory = 'data/resized'

# Results Main Directory
results_dir = 'results'

# Segmentation Based Model Directory
segmentation_based_model_results_dir = os.path.join(results_dir, 'segmentation-based-model')
if os.path.isdir(segmentation_based_model_results_dir) == False:
        os.mkdir(segmentation_based_model_results_dir)

# Segmentation Based Model U-Net++ Directory
segmentation_based_model_unet_pp_dir = os.path.join(segmentation_based_model_results_dir, 'unet-pp')
if os.path.isdir(segmentation_based_model_unet_pp_dir) == False:
        os.mkdir(segmentation_based_model_unet_pp_dir)

# Segmentation Based Model U-Net++ Weights Directory
segmentation_based_model_unet_pp_weights_dir = os.path.join(segmentation_based_model_unet_pp_dir, 'weights')
if os.path.isdir(segmentation_based_model_unet_pp_weights_dir) == False:
        os.mkdir(segmentation_based_model_unet_pp_weights_dir)

# Segmentation Based Model U-Net++ Predicted Masks Directory
segmentation_based_model_unet_pp_masks_dir = os.path.join(segmentation_based_model_unet_pp_dir, 'predictions')
if os.path.isdir(segmentation_based_model_unet_pp_masks_dir) == False:
    os.mkdir(segmentation_based_model_unet_pp_masks_dir)

# Open X data (images)
# X_train data
with open(os.path.join(data_directory, 'X_train_221.pickle'), 'rb') as fp: 
        X_train = cPickle.load(fp)

# X_test data
with open(os.path.join(data_directory, 'X_test_221.pickle'), 'rb') as fp: 
        X_test = cPickle.load(fp)

# Concatenate both to obtain full X data
X = np.concatenate((X_train, X_test))

# X data preprocessing
X = np.array(X, dtype='float')


# Open test indices list
with open('data/train-test-indices/test_indices_list.pickle', 'rb') as f:
    test_indices_list = cPickle.load(f)


# Iterate through folds
for fold in FOLDS:
    print('Current fold: {}'.format(fold))

    # Clear Keras Session to avoid RAM-memory problems
    K.clear_session()

    # Create a model object
    model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
    model.compile('Adadelta', 'binary_crossentropy', ['binary_accuracy'])   
    # model.summary()
    
    # Load model weights
    print("Loading model weights...")
    weights = os.path.join(segmentation_based_model_unet_pp_weights_dir, 'unet_pp_ADADELTA_CV5_Fold_{}.hdf5'.format(fold))
    model.load_weights(weights)
    print("Model weights loaded.")

    # Fold X_data
    X_fold_data = X[test_indices_list[fold]]

    # Start time to measure algorithm execution performance
    startTime = time.time()

    # Generate U-Net++ predictions (U-Net++ predicted masks)
    predictions = model.predict(X_fold_data, batch_size=1, verbose=True)

    # Elapsed time computation
    print ('The script took {} seconds for fold {}!'.format(time.time() - startTime, fold))

    # Save U-Net++ predictions file (masks)
    with open(os.path.join(segmentation_based_model_unet_pp_masks_dir, 'unet_pp_preds_CV5_Fold_{}.pickle'.format(fold)), 'wb') as fp:
        cPickle.dump(predictions, fp, -1)

print('Segmentation Based Model U-Net++ Masks Prediction finished.')