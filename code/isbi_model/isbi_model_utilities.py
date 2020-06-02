# Imports
import _pickle as cPickle
import numpy as np
import cv2

# Keras Imports 
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, multiply, concatenate, Dense, Flatten, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from keras import losses
from keras import backend as K

# Utility functions
# The VGG16 Backbone 
def create_conv_base(rows=384, columns=512, channels=3):
    conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (rows, columns, channels))
    
    conv_base.trainable=True 
    
    set_trainable = False
    
    for layer in conv_base.layers: 
        if layer.name=='block5_conv1' or layer.name=='block5_conv2' or layer.name=='block5_conv3' or layer.name=='block4_conv1' or layer.name=='block4_conv2' or layer.name=='block4_conv3':
            set_trainable = True
        if set_trainable == True: 
            layer.trainable = True
        else:
            layer.trainable = False

    return conv_base

# U-Net
def u_net(inputs): 
    u_net_input = Lambda(lambda inputs:(inputs/255.0))(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(u_net_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    
    return conv10

# Module for feature extraction
def feature_extraction(inputs):
    conv_base = create_conv_base()
    conv1 = conv_base(inputs)
    conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
    conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
    conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
    conv1 = Conv2D(512, (1, 1), activation='relu', padding='valid')(conv1)
    flat = Flatten()(conv1)
    dense1 = Dense(256,activation='relu')(flat)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(128,activation='relu')(dense1)
    reg = Dense(74,activation='sigmoid', name = 'keypoints')(dense1)
        
    return reg



# Function that creates the model
def create_isbi_model(weights=False, rows=384, columns=512, channels=3):
    K.clear_session()

    # Input is an image with 3 colour channels
    inputs = Input((rows, columns, 3))

    # First step is to obtain probability maps
    stage1 = u_net(inputs)
    
    #Concatenate probability maps in order to have an image with the same number of channels as input image 
    stage1_concat = concatenate([stage1,stage1,stage1])
    
    # Multiplication between prob maps and input image, to select region of interest
    stage2_in = multiply([stage1_concat,inputs])
    
    stage2 = u_net(stage2_in)
    
    stage2_concat = concatenate([stage2,stage2,stage2])
    
    stage3_in = multiply([stage2_concat,inputs])
    
    stage3 = u_net(stage3_in)
    
    stage3_concat = concatenate([stage3, stage3, stage3])
    
    stage4_in = multiply([stage3_concat, inputs])
    
    # Perform regression
    stage4 = feature_extraction(stage4_in)
    
    
    # Model has one input: image; and two outputs: probability maps and keypoints
    model = Model(inputs=[inputs], outputs=[stage1, stage2, stage3, stage4])

    model.compile(optimizer='adadelta', loss = [losses.mean_squared_error, losses.mean_squared_error, losses.mean_squared_error, losses.mean_squared_error], loss_weights= [1, 2, 4, 10])

    if weights:
        print('Loading model weights...')
        model.load_weights(weights)
        print('Model weights loaded.')

    
    return model

# Function to generate predictions
def generate_isbi_predictions(X_train_path, X_test_path, test_indices_path, fold, isbi_model_weights_path):

    # Open X_train data
    with open(X_train_path, 'rb') as f:
        X_train = cPickle.load(f)

    # Open X_test data
    with open(X_test_path, 'rb') as p:
        X_test = cPickle.load(p)

    # Create complete X dataset
    X = np.concatenate((X_train, X_test))
    
    # Open test indices and fold
    with open(test_indices_path, 'rb') as f:
        test_indices_list = cPickle.load(f)

    # Obtain "fold" dataset
    X = X[test_indices_list[fold]]

    # X data preprocessing
    X = np.array(X, dtype='float')
    X = preprocess_input(X)

    # Create model and load weights
    model = create_isbi_model(weights=isbi_model_weights_path)

    # Generate predictions
    predictions = model.predict(X, batch_size=1, verbose=True)[3]

    return predictions