# Imports
import numpy as np 
import _pickle as cPickle
import os 

# Tensorflow Imports
import tensorflow as tf

# Tensorflow trick to solve some compatibility issues 
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Keras Imports
from keras import backend as K

# Segmentation Based Model Imports
from code.segmentation_based_model.unet_plus_plus_model.segmentation_model_train_generator import Generator
from code.segmentation_based_model.unet_plus_plus_model.segmentation_models import Xnet

# CUDA Environment Variables (adapt them to your personal settings)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# GLOBAL VARIABLES
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

# Open X data (images)
# X_train data
with open(os.path.join(data_directory, 'X_train_221.pickle'), 'rb') as fp: 
        X_train = cPickle.load(fp)

# X_test data
with open(os.path.join(data_directory, 'X_test_221.pickle'), 'rb') as fp: 
        X_test = cPickle.load(fp)

# Concatenate both to obtain full X data
X = np.concatenate((X_train, X_test))


# Open masks data (masks)
# mask_train data
with open(os.path.join(data_directory, 'mask_train_221.pickle'), 'rb') as fp: 
        mask_train = cPickle.load(fp)

# mask_test data
with open(os.path.join(data_directory, 'mask_test_221.pickle'), 'rb') as fp: 
        mask_test = cPickle.load(fp)

# Concatenate both to obtain full masks data
masks = np.concatenate((mask_train, mask_test))

# Open train indices list
with open('data/train-test-indices/train_indices_list.pickle', 'rb') as fp:
        train_indices_list = cPickle.load(fp)


# Data preprocessing
# X data
X = np.array(X, dtype='float')

# masks data
masks = np.array(masks) / np.max(masks)
masks = masks.reshape((-1, 384, 512, 1))

# print(np.min(masks), np.max(masks))

# Iterate through all folds (in train indices list)
for index in range(np.shape(train_indices_list)[0]):
        print('Current fold: {}'.format(index))

        # Clear Keras Session to avoid RAM-memory problems
        K.clear_session()

        # Create model object
        model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
        model.compile('Adadelta', 'binary_crossentropy', ['binary_accuracy'])
        # model.summary()

        # Create a Generator object to feed the model
        my_generator = Generator(
                X[train_indices_list[index]],
                masks[train_indices_list[index]],
                batchsize=2,
                flip_ratio=0.5,
                translation_ratio=0.5,
                rotate_ratio = 0.5
                )

        # Train Model
        model.fit_generator(my_generator.generate(), epochs=300, steps_per_epoch=my_generator.size_train/2, verbose=2)

        # Save model weights
        model.save_weights(os.path.join(segmentation_based_model_unet_pp_weights_dir, 'unet_pp_ADADELTA_CV5_Fold_{}.hdf5'.format(index)))