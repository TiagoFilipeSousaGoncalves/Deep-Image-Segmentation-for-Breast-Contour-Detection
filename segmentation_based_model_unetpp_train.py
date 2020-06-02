import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import _pickle as cPickle
import numpy as np 
from train_generator_mask import Generator

from keras import backend as K

""" Load all files """
with open("/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/X_train_221.pickle",'rb') as fp: 
        X_train = cPickle.load(fp)

with open("/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/X_test_221.pickle",'rb') as fp: 
        X_test = cPickle.load(fp)

X = np.concatenate((X_train, X_test))

with open("/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/mask_train_221.pickle",'rb') as fp: 
        mask_train = cPickle.load(fp)

with open("/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/mask_test_221.pickle",'rb') as fp: 
        mask_test = cPickle.load(fp)

masks = np.concatenate((mask_train, mask_test))


with open('/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/train_indices_list.pickle', 'rb') as fp:
        train_indices_list = cPickle.load(fp)

with open('/data/tgoncalv/GitLab/deep-image-segmentation-for-breast-contour-detection/test_indices_list.pickle', 'rb') as fp:
        test_indices_list = cPickle.load(fp)




X = np.array(X, dtype='float')
masks = np.array(masks) / np.max(masks)
masks = masks.reshape((-1, 384, 512, 1))

print(np.min(masks), np.max(masks))

from segmentation_models import Unet, Nestnet, Xnet

# prepare data
#x, y = ... # range in [0,1], the network expects input channels of 3

for index in range(np.shape(train_indices_list)[0]):
        print('Current fold {}'.format(index))

        K.clear_session()
        # prepare model
        model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
        # model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
        # model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

        #model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
        model.compile('Adadelta', 'binary_crossentropy', ['binary_accuracy'])
        model.summary()

        my_generator = Generator(X[train_indices_list[index]],
                                masks[train_indices_list[index]],
                                batchsize=2,
                                flip_ratio=0.5,
                                translation_ratio=0.5,
                                rotate_ratio = 0.5)

        # train model
        #model.fit(x=X_train, y=mask_train, batch_size=4, epochs=100, verbose=1, validation_split=0.2)
        model.fit_generator(my_generator.generate(), epochs=300, steps_per_epoch=my_generator.size_train/2, verbose=2)
        model.save_weights('unet_pp_ADADELTA_CV5_Fold_{}.hdf5'.format(index))