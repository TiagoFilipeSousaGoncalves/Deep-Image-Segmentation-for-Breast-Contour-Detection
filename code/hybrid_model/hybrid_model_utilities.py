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
from priodict import priorityDictionary
from DijkstraPaths import *
from shortest_paths import *


# Hybrid Model Functions
# Compute Breast Priors Functions
# Create Mask Function
def create_mask(all_shapes):
    final_mask = np.zeros([1800, 2100])
    # final_mask = np.zeros([1000, 2000])
    for y in all_shapes:
        y[:, 0] *= (-1)
        y *= 1000
        y += [100, 500]
        y = np.array(spline(y)).transpose()
        for point in y:
            final_mask[tuple(point.astype(int))] = 1

    final_mask = morpho.binary_fill_holes(final_mask)
    b = morpho.binary_erosion(final_mask, structure=np.ones([3, 3]))
    # plt.imshow(b)
    # plt.show()
    x, y = np.nonzero(np.logical_and(final_mask, np.logical_not(b)))
    points = np.stack((x, y), axis=1).astype(float)
    points -= [100, 500]
    points /= 1000
    return points

# Create Breast Contour Priors Function
def save_breast_contour_params(y_train_path, y_test_path, fold, train_indices_list_path, breast_prior_save_path):
    # Open y_train data
    f = open(y_train_path, 'rb')
    y_train = pkl.load(f)
    f.close()

    # Open y_test data
    f = open(y_test_path, 'rb')
    y_test = pkl.load(f)
    f.close()
    
    # Open train_indices_list
    with open(train_indices_list_path, 'rb') as f:
        train_indices_list = pkl.load(f)
    
    # Concatenate and access fold to choose the correct indices
    Y = np.concatenate((y_train, y_test))
    Y = Y[train_indices_list[fold]]
    Y = np.array(Y, dtype='float')
    
    # Apply transformations to obtain breast contour priors
    all_shapes = list()
    for curr_y in Y:
        for side in ["left", "right"]:
            if side is "left":
                y = curr_y[0:34]
                y = np.reshape(y, [-1, 2])
            else:
                y_ = curr_y[34:68]

                y = np.reshape(y_, [-1, 2])
                y[:, 0] *= (-1)

            y = transform(y)
            all_shapes.append(y)

    points = create_mask(all_shapes)

    # Save breast contour prior
    # Create filename
    filename = os.path.join(breast_prior_save_path, 'breast_contour_prior_CV_Fold_{}.pkl'.format(fold))
    
    # Save pickle file
    with open(filename, "wb") as f:
        pkl.dump(points, f, -1)


"""
_____________________________ AUXILIARY FUNCTIONS _____________________________

"""

# Gradient Image Function
def gradient_mag(img):
    # Returns the gradient magnitude of the input image
    gx = skimage.filters.sobel_h(img)
    gy = skimage.filters.sobel_v(img)
    magnitude = np.sqrt(gx**2+gy**2)
    return magnitude

# Harris Corner Measure Function
def harris_corner_measure(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 5, 3, 0.04)
    return dst

# Create Breast Mask Function
def create_breast_mask(shape, points, left=True):
    mask = np.zeros(shape)
    y = np.array(spline(points)).transpose()
    y = y.round().astype(int)
    y[:, 0] = np.clip(y[:, 0], 0, shape[0]-1)
    y[:, 1] = np.clip(y[:, 1], 0, shape[1]-1)
    mask[y[:, 0], y[:, 1]] = 1
    hor = np.linspace(y[0, 0], y[-1, 0], 1000).round().astype(int)
    ver = np.linspace(y[0, 1], y[-1, 1], 1000).round().astype(int)
    mask[hor, ver] = 1
    return morpho.binary_fill_holes(mask)

# Spline Function 
def spline(points, n_points=10000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:, 0]
    #x = points[0:72:2]
    y = points[:, 1]
    #y = points[1:73:2]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

# Circular Path Features Function
def circular_path_features(img, point):
    polar, mapping = topolar(img, point, R, ret_mapping=True)
    final = np.abs(scipy.signal.convolve2d(polar, filt2, "valid"))
    final2 = final / final.max() * 255

    paths = shortest_paths(final2, beta=0.02)[0]
    indexes = np.nonzero(np.abs(paths[:, 0, 1]-paths[:, 719, 1]) <= 1)
    polar_path = paths[indexes[0][0], :, :]

    mean_intensity = np.mean(final[polar_path[:, 0], polar_path[:, 1]])

    for point in range(polar_path.shape[0]):
        polar_path[point] = mapping(polar_path[point])

    poly = Polygon(polar_path)
    A = poly.area
    P = poly.length
    Sa = (4*np.pi*A)/P**2
    da = np.sqrt((4*A)/np.pi)

    return [mean_intensity, Sa, da]

# Find Breast Contour Function
def find_breast_contour(g_mag, p1, p2, PRIOR_POINTS):
    # Adjust the contour prior to the position of this specific image
    prior_points = adjust(PRIOR_POINTS.copy(), p1, p2)
    prior_points[:, 0] = np.clip(prior_points[:, 0], 0, g_mag.shape[0]-1)
    prior_points[:, 1] = np.clip(prior_points[:, 1], 0, g_mag.shape[1]-1)

    # Compute the prior mask
    prior = np.zeros(g_mag.shape)
    prior[prior_points[:, 0].astype(int), prior_points[:, 1].astype(int)] = 1
    prior = morpho.binary_fill_holes(prior)
    prior = morpho.distance_transform_edt(1-prior)
    # plt.imshow(g_mag)
    # plt.show() 
    # print("GMAG: ", np.max(g_mag))
    # plt.imshow(prior)
    # plt.show()

    # Compute the shortest path
    distance_funcs = g_mag, prior
    G = build_graph(distance_funcs, [0, 0, *prior.shape])
    path = shortestPath(G, tuple(p1.astype(int)), tuple(p2.astype(int)))
    path = np.asarray(path)
    return path

# Generate Hybrid Predictions Function
def generate_hybrid_predictions(X_train_path, X_test_path, test_indices_list_path, fold, keypoints, breast_prior_path, result_path):
    # Open breast contour prior
    with open(breast_prior_path, "rb") as fp:
        PRIOR_POINTS = pkl.load(fp)

    # Open X_train
    with open(X_train_path, "rb") as fp:
        X_train = pkl.load(fp)

    # Open X_test
    with open(X_test_path, "rb") as fp:
        X_test = pkl.load(fp)

    # Concatenate both and access the rigt fold-indices
    X = np.concatenate((X_train, X_test))

    with open(test_indices_list_path, 'rb') as f:
        test_indices_list = pkl.load(f)

    X = X[test_indices_list[fold]]

    # Find Predictions
    # Number of images
    N = X.shape[0]

    # Create a list to append hybrid-model predictions
    all_hybrid_predictions = []

    # For each image in the dataset
    for i in range(N):
        print("Image {} of {}".format(i+1, N))

        # Create a temporary list to append iteration predictions
        hybrid_preds = []

        # Access image
        img = X[i]
        img_grey = np.average(img, axis=2)

        # Compute gradient magnitude
        g_mag = gradient_mag(img_grey)

        # Endpoints: left, mid and right: Given by DNN!
        lp = keypoints[i][0:2]
        lmp = keypoints[i][32:34]
        lp = np.flip(lp)
        lmp = np.flip(lmp)

        rp = keypoints[i][34:36]
        rmp = keypoints[i][66:68]

        # Need to do Flip Trick to Correctly Compute Right Contour Shortest-Path
        rp = np.flip(keypoints[i][34:36])
        rmp = np.flip(keypoints[i][66:68])

        # Find the left breast countour
        left_contour = find_breast_contour(g_mag, lp, lmp, PRIOR_POINTS)

        # Obtain 17 equaly spaced points (keypoints)
        jump = (left_contour.shape[0]-1)/16
        indexes = [int(x*jump) for x in range(17)]
        left_contour = left_contour[indexes, :]


        # Find the right breast countour
        right_contour = find_breast_contour(g_mag, rmp, rp, PRIOR_POINTS)

        # obtain 17 equaly spaced points (keypoints)
        jump = (right_contour.shape[0]-1)/16
        indexes = [int(x*jump) for x in range(17)]
        right_contour = right_contour[indexes, :]

        # Find the nipple position for each breast: DNN Nipples Predictions
        # Left Nipple
        nip_l = keypoints[i][70:72]
        
        # Right Nipple
        nip_r = keypoints[i][72:74]

        #Sternal Notch
        stn = keypoints[i][68:70]

        # Add breast contour keypoints, midpoints and nipple key-points to pred. list
        hybrid_preds.append(left_contour[:, ::-1])
        # hybrid_preds.append(lmp[::-1])

        hybrid_preds.append(right_contour[:, ::-1])
        # hybrid_preds.append(rmp[::-1])

        hybrid_preds.append(stn[::-1])
        hybrid_preds.append(nip_l[::-1])
        hybrid_preds.append(nip_r[::-1])


        # Add predictions list to the complete results
        # print("Hybrid Predictions Shape is: ", np.shape(hybrid_preds))
        all_hybrid_predictions.append(hybrid_preds)

    # Save results
    all_hybrid_predictions = np.array(all_hybrid_predictions)
    # print(all_hybrid_predictions.shape)
    filename = os.path.join(result_path, "hybrid_model_preds_CV_Fold_{}.pickle".format(fold))
    with open(filename, "wb") as fp:
        pkl.dump(all_hybrid_predictions, fp, -1)


# Hybrid Model Predictions Reshape Functions
def reshape_hybrid_predictions(hybrid_model_raw_predictions):
    # Create list to append predictions
    hybrid_predictions = []

    # Iterate through the raw hybrid model predictions
    for index in range(hybrid_model_raw_predictions.shape[0]):
        tmp = []

        # Left Contour
        for value in hybrid_model_raw_predictions[index][0]:
            tmp.append(value[0])
            tmp.append(value[1])

        # Right Contour
        for value in hybrid_model_raw_predictions[index][1]:
            tmp.append(value[0])
            tmp.append(value[1])
    
        # Endpoints
        tmp.append(hybrid_model_raw_predictions[index][2][1])
        tmp.append(hybrid_model_raw_predictions[index][2][0])
    
        
        tmp.append(hybrid_model_raw_predictions[index][3][1])
        tmp.append(hybrid_model_raw_predictions[index][3][0])

        tmp.append(hybrid_model_raw_predictions[index][4][1])
        tmp.append(hybrid_model_raw_predictions[index][4][0])

        hybrid_predictions.append(tmp)

    hybrid_predictions = np.array(hybrid_predictions)
    
    # print(hybrid_predictions.shape)
    # print(hybrid_predictions[0])


    return hybrid_predictions