import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

import numpy as np 
import _pickle as cPickle

from segmentation_models import Unet, Nestnet, Xnet
import time 

# prepare data
#x, y = ... # range in [0,1], the network expects input channels of 3

# prepare model
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++
# model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
# model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

model.compile('Adadelta', 'binary_crossentropy', ['binary_accuracy'])
model.summary()

fold = 4

model.load_weights('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/CV5_model_UNetPP_29nov2019/unet_pp_ADADELTA_CV5_Fold_{}.hdf5'.format(fold))

with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/X_train_221.pickle', 'rb') as fp:
    X_train = cPickle.load(fp)

with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/X_test_221.pickle', 'rb') as fp:
    X_test = cPickle.load(fp)


with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/test_indices_list.pickle', 'rb') as f:
    test_indices_list = cPickle.load(f)

X = np.concatenate((X_train, X_test))
X = X[test_indices_list[fold]]

X = np.array(X, dtype='float')

with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/mask_train_221.pickle', 'rb') as fp:
    mask_train = cPickle.load(fp)

with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/mask_test_221.pickle', 'rb') as fp:
    mask_test = cPickle.load(fp)


mask = np.concatenate((mask_train, mask_test))
mask = mask[test_indices_list[fold]]

mask = np.array(mask) / np.max(mask)
mask = mask.reshape((-1, 384, 512, 1))

print(np.max(mask), np.min(mask))

#evaluation = model.evaluate(X, mask, batch_size=1, verbose=True)
#print(evaluation)
"""No Aug: [Loss, Acc] = [0.13015010908468447, 0.9821957349777222]"""
"""W/ Aug: [Loss, Acc] = [0.08433691901502324, 0.9784207344055176]"""
"""W/Aug and ADADELTA: [Loss, Acc] = [0.12168492265601656, 0.9858260750770569]"""

startTime = time.time()
predictions = model.predict(X, batch_size=1, verbose=True)
print ('The script took {} second !'.format(time.time() - startTime))

with open('unet_pp_preds_CV5_Fold_{}.pickle'.format(fold), 'wb') as fp:
    cPickle.dump(predictions, fp, -1)