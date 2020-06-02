#Scoring
from shapely.geometry import LineString, Point
import numpy as np
import scipy.interpolate as interpolate
import pickle 
import sys
import os

import matplotlib.pyplot as plt 

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


def compute_euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))
    
def get_curves_distance(points_a,points_b,n_points=5000):
    points_a = np.asarray(points_a).reshape([-1,2])
    points_b = np.asarray(points_b).reshape([-1,2])

    distance = curve_dist_aux(points_a,points_b,n_points)
    distance+= curve_dist_aux(points_b,points_a,n_points)
    distance/=2
    return distance

def curve_dist_aux(curve_points,points,n_points=5000):
    curve = spline(curve_points,n_points)
    curve = np.stack(curve,axis=1)
    curve = LineString(curve)
    distance = 0
    for point in points:
        distance+=curve.distance(Point(point))
    distance/=len(points)
    return distance

def spline(points,n_points=5000):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    #Minor Bug Fix On Interpolation
    #Check: https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs
    #okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    #xp = np.r_[x[okay], x[-1], x[0]]
    #xp = np.r_[x[okay], x[-1]]
    #yp = np.r_[y[okay], y[-1], y[0]]
    #yp = np.r_[y[okay], y[-1]]

    #tck, u = interpolate.splprep([xp, yp], s=0)
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

def scoring(predictions, y, img_shape, dataset="221"):

    diagonal_dataset120 = 2549.7938
    diagonal_dataset221 = 2701.6085
    
    score = []
    
    diagonal = img_shape[0:2]
    diagonal = np.sqrt(diagonal[0]**2+diagonal[1]**2)

    if dataset == "120":
        normalize = diagonal_dataset120
    elif dataset == "221":
        normalize = diagonal_dataset221
    #else:
    #    normalize = diagonal
    
    
    truth_lp = y[0:2]
    truth_midl = y[32:34]
    truth_midr = y[66:68]
    truth_rp = y[34:36]
    
    
    truth_l_breast = y[0:34]
    truth_r_breast = y[34:68]

    #truth_breasts = y[0:68]
    
    truth_l_nipple = y[70:72]
    #truth_l_nipple = y[68:70]

    truth_r_nipple = y[72:74]
    #truth_r_nipple = y[70:72]

    #truth_l_nipple = y[68:70]
    #truth_r_nipple = y[70:72]

    score.append(compute_euclidean_distance(predictions[0], truth_lp) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[1], truth_midl) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[2], truth_midr) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[3], truth_rp) * (normalize / diagonal))    

    score.append(get_curves_distance(np.array(predictions[4], dtype='float64'), np.array(truth_l_breast, dtype='float64')) * (normalize / diagonal))
    score.append(get_curves_distance(np.array(predictions[5], dtype='float64'), np.array(truth_r_breast, dtype='float64')) * (normalize / diagonal))
    
    #Change this every time you change evaluation method
    #score.append(get_curves_distance(np.array(predictions[4], dtype='float64'), np.array(truth_breasts, dtype='float64') * (normalize / diagonal)))
    #score.append(compute_euclidean_distance(predictions[5], truth_l_nipple) * (normalize / diagonal))
    #score.append(compute_euclidean_distance(predictions[6], truth_r_nipple) * (normalize / diagonal))
    
    score.append(compute_euclidean_distance(predictions[6], truth_l_nipple) * (normalize / diagonal))
    score.append(compute_euclidean_distance(predictions[7], truth_r_nipple) * (normalize / diagonal))
    return score

def dense_scoring(scores):
    
    scores = np.asarray(scores)
    end_point_scores = scores[:,0:4]
    end_points = [np.mean(end_point_scores),
                  np.std(end_point_scores),
                  np.max(end_point_scores)
                  ]
    breast_contour_scores = scores[:,4:6]
    #breast_contour_scores = scores[:,4]
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

def to_strange_fmt(y, mode='mixed'):
    y = np.asarray(y, dtype='float64')
    #if seg_contour:
    #  seg_contour = np.array(seg_contour, dtype='float64')
    
    lpredictions = []
    lpredictions.append(y[0:2])
    lpredictions.append(y[32:34])
    lpredictions.append(y[66:68])
    lpredictions.append(y[34:36])
    
    lpredictions.append(y[0:34])
    lpredictions.append(y[34:68])
    #lpredictions.append(y)

    #if seg_contour:
    #  lpredictions.append(seg_contour)
    
    if mode == 'hybrid':
        lpredictions.append(y[68:70])
        lpredictions.append(y[70:72])

    elif mode == 'mixed':
        lpredictions.append(y[70:72])
        lpredictions.append(y[72:74])

    return lpredictions


MODELS = ['UNET', 'UNET_HF', 'GCN', 'MIXED', 'Hybrid', 'ISBI']
model = MODELS[4] #0 for U-Net  or 1 for GCN


if model == 'UNET':
    with open('code/U_Net/predictions/unet_pred_kpts.pickle', 'rb') as f:
        pred_kpts = pickle.load(f)
    pred_kpts *= 512

elif model == 'UNET_HF':
    with open('code/U_Net/predictions/unet_pred_kpts_half_filters.pickle', 'rb') as f:
        pred_kpts = pickle.load(f)
    pred_kpts *= 512

elif model == 'GCN':
    with open('code/GCN/predictions/gcn_pred_kpts.pickle', 'rb') as f:
        pred_kpts = pickle.load(f)
    pred_kpts *= 512


elif model == 'MIXED':
    fold = 4
    with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/mixed_preds_CV_Fold_{}.pickle'.format(fold), 'rb') as f:
        pred_kpts = pickle.load(f)

elif model == 'Hybrid':
    fold = 4
    with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/hybrid_preds_CV_Fold_{}.pickle'.format(fold), 'rb') as f:
        pred_kpts = pickle.load(f)

elif model == 'ISBI':
    fold = 4
    with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/isbi_preds_w_only_CV_Fold_{}.pickle'.format(fold), 'rb') as f:
        pred_kpts = pickle.load(f)
    
    pred_kpts = np.array(pred_kpts)
    pred_kpts *= 512


    print(pred_kpts)

#with open("/gdrive/My Drive/New Segmentation Test/New Architecture/Wilson Files/120dataset/Original Files/X_test_120.pickle",'rb') as fp:
with open("/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/original_files/X_train_221.pickle",'rb') as fp: 
        X_train_original = pickle.load(fp) 


with open("/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/original_files/X_test_221.pickle",'rb') as fp: 
        X_test_original = pickle.load(fp)  

with open('/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/test_indices_list.pickle', 'rb') as f:
            test_indices_list = pickle.load(f)

X_original = np.concatenate((X_train_original, X_test_original))

X_original = X_original[test_indices_list[fold]]



resized_preds = resize_keypoints_to_original_size(pred_kpts, X_original.copy())
#resized_contours = resize_keypoints_to_original_size(final_contours, X_test_original)


with open("/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/original_files/y_train_221.pickle",'rb') as fp:  
        y_train_original = pickle.load(fp)  

with open("/home/ctm/Desktop/GitLab/deep-image-segmentation-for-breast-contour-detection/code/Wilson_Files_Data/original_files/y_test_221.pickle",'rb') as fp:  
        y_test_original = pickle.load(fp)  
        
y_original = np.concatenate((y_train_original, y_test_original))
y_original = y_original[test_indices_list[fold]]





predictions = [] 

for i in range(np.shape(X_original)[0]):
    predictions.append(to_strange_fmt(resized_preds[i]))
    #predictions.append(to_strange_fmt(resized_contours[i]))
    #predictions.append(to_strange_fmt(resized_preds[i], resized_contours[i]))


scores = [] 

for i in range(np.shape(X_original)[0]): 
    scores.append(scoring(predictions[i], y_original[i], X_original[i].shape, dataset="221"))

print('For model: {}\n'.format(model), dense_scoring(scores))


for i in range(X_original.shape[0]):
    plt.imshow(X_original[i], cmap='gray')
    plt.plot(resized_preds[i][0:74:2], resized_preds[i][1:75:2], 'o')
    plt.show()