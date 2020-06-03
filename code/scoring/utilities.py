# Imports
# Generic Imports
from shapely.geometry import LineString, Point
import numpy as np
import scipy.interpolate as interpolate
import _pickle as cPickle 
import sys
import os
import matplotlib.pyplot as plt 

# Scoring Functions
# Euclidean Distance
def compute_euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

# Helper Function
def get_curves_distance(points_a, points_b, n_points=5000):
    points_a = np.asarray(points_a).reshape([-1,2])
    points_b = np.asarray(points_b).reshape([-1,2])

    distance = curve_dist_aux(points_a,points_b,n_points)
    distance+= curve_dist_aux(points_b,points_a,n_points)
    distance/=2
    return distance

# Helper Function
def curve_dist_aux(curve_points, points, n_points=5000):
    curve = spline(curve_points,n_points)
    curve = np.stack(curve,axis=1)
    curve = LineString(curve)
    distance = 0
    for point in points:
        distance+=curve.distance(Point(point))
    distance/=len(points)
    return distance

# Spline Function
def spline(points, n_points=5000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]

    # Minor Bug Fix On Interpolation, check this if you are getting errors
    # Check: https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    # okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    # xp = np.r_[x[okay], x[-1], x[0]]
    # xp = np.r_[x[okay], x[-1]]
    # yp = np.r_[y[okay], y[-1], y[0]]
    # yp = np.r_[y[okay], y[-1]]

    # tck, u = interpolate.splprep([xp, yp], s=0)
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

# Scoring Function
def scoring(predictions, y, img_shape, dataset_diagonal, dataset=False):
    # Diagonals per dataset
    # Dataset '120'
    diagonal_dataset120 = 2549.7938
    # Dataset '221'
    diagonal_dataset221 = 2701.6085
    
    score = []
    
    diagonal = img_shape[0:2]
    diagonal = np.sqrt(diagonal[0]**2+diagonal[1]**2)

    if dataset == "120":
        normalize = diagonal_dataset120
    elif dataset == "221":
        normalize = diagonal_dataset221
    else:
        normalize = dataset_diagonal
    
    # Ground-Truth Endpoints
    truth_lp = y[0:2]
    truth_midl = y[32:34]
    truth_midr = y[66:68]
    truth_rp = y[34:36]
    
    # Ground-Truth Breast Contours
    truth_l_breast = y[0:34]
    truth_r_breast = y[34:68]
    
    # Ground-Truth Nipples
    truth_l_nipple = y[70:72]
    truth_r_nipple = y[72:74]

    # Compute Endpoints Scores
    score.append(compute_euclidean_distance(predictions[0], truth_lp) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[1], truth_midl) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[2], truth_midr) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[3], truth_rp) * (normalize / diagonal))    

    # Compute Breast Contour Scores
    score.append(get_curves_distance(np.array(predictions[4], dtype='float64'), np.array(truth_l_breast, dtype='float64')) * (normalize / diagonal))
    score.append(get_curves_distance(np.array(predictions[5], dtype='float64'), np.array(truth_r_breast, dtype='float64')) * (normalize / diagonal))
    
    # Compute Nipples Scores
    score.append(compute_euclidean_distance(predictions[6], truth_l_nipple) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[7], truth_r_nipple) * (normalize / diagonal))
    
    return score

# Final Scoring Function
def dense_scoring(scores):
    
    scores = np.asarray(scores)
    end_point_scores = scores[:,0:4]
    end_points = [np.mean(end_point_scores),
                  np.std(end_point_scores),
                  np.max(end_point_scores)
                  ]
    
    breast_contour_scores = scores[:,4:6]
    breast_contour = [np.mean(breast_contour_scores),
                      np.std(breast_contour_scores),
                      np.max(breast_contour_scores)
                      ]
    
    nipple_scores = scores[:,6:8]
    nipple = [np.mean(nipple_scores),
              np.std(nipple_scores),
              np.max(nipple_scores)
              ]
    
    return end_points, breast_contour, nipple

# Create arrays of predictions
def to_strange_fmt(y):
    y = np.asarray(y, dtype='float64')
    
    lpredictions = []

    # Append Endpoints
    lpredictions.append(y[0:2])
    lpredictions.append(y[32:34])
    lpredictions.append(y[66:68])
    lpredictions.append(y[34:36])
    
    # Append Breast Contours
    lpredictions.append(y[0:34])
    lpredictions.append(y[34:68])

    # Append Nipples
    lpredictions.append(y[70:72])
    lpredictions.append(y[72:74])

    return lpredictions

# Resize Keypoints to Original Size Function
def resize_keypoints_to_original_size(keypoint_predictions, X_original):
    X_original = np.array(X_original)
    keypoint_predictions = np.array(keypoint_predictions).copy()
    
    final_predictions = [] 

    for i in range(X_original.shape[0]):
        rows, columns, channels = np.shape(X_original[i])
        x1 = rows / 1536
        x2 = columns / 2048
        for j in range(keypoint_predictions[i].shape[0]): 
            if(j % 2 == 0):
                keypoint_predictions[i][j] *= x2
            else: 
                keypoint_predictions[i][j] *= x1
        final_predictions.append(keypoint_predictions[i] * 4)
        
    return final_predictions