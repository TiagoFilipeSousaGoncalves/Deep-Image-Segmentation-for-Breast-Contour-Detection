"""
Created on Mon Jul 16 13:53:37 2018

@author: eduardo

Shortest path functions
"""
import numpy as np
from scipy.ndimage.interpolation import geometric_transform

alpha = 0.15
beta = 0.08
delta = 1.85
MAX_GRADIENT = 255


def shortest_paths(g_mag, alpha=alpha, beta=beta, delta=delta):
    ai, aj = g_mag.shape

    dists = np.zeros([aj])
    paths = [[] for y in range(aj)]

    for i in range(1, ai):
        curr_line_dists = np.zeros([aj])
        curr_line_paths = []
        for j in range(aj):
            start = max(0, j-1)
            end = min(aj, j+2)
            coords = np.meshgrid(i-1, np.arange(start, end))
            coords = np.array(coords).T.reshape([-1, 2])
            pos = np.array([i, j])

            geometric_dist = np.sqrt(np.sum((coords-pos)**2, axis=1))
            min_gradient = np.minimum(g_mag[tuple(pos)], g_mag[i-1, start:end])
            gradient_distance = alpha*np.exp(beta*(MAX_GRADIENT-min_gradient)) + delta

            neigh_dists = dists[start:end] + geometric_dist * gradient_distance

            selected = np.argmin(neigh_dists)
            curr_line_dists[j] = neigh_dists[selected]
            curr_line_paths.append([tuple(coords[selected])])

        dists = curr_line_dists

        temp_paths = []
        for j in range(aj):
            temp_paths.append(curr_line_paths[j]+paths[curr_line_paths[j][0][1]])
        paths = temp_paths

    temp_paths = []
    for j in range(aj):
        temp_paths.append([(ai-1, j)]+paths[j])
    paths = temp_paths
    paths = np.array(paths)

    return paths, dists


def get_mapping_function(C, ao, ar, maxo, maxr, r_min=20):
    # Returns a function which can be used to transform cartesian coordinates
    # to polar ones
    # C: center of coordinates
    # ao, ar: polar image size
    # maxo, maxr: maximum omega and radius

    def func(coords):
        x, y = coords

        o = (x / ao) * maxo
        r = r_min + (y / ar) * (maxr - r_min)

        new_x = r * np.cos(o)
        new_y = r * np.sin(o)

        return new_x+C[0], new_y+C[1]

    return func


def topolar(img, C, R, order=1, ret_mapping=False):
    # Transforms an image from cartasian coordinates to polar coordinates
    # img: image
    # C: center of polar coordinates
    # R max radius
    # order: order of the spline interpolation

    mapping_function = get_mapping_function(C, 720, 200, 2*np.pi, R)
    polar = geometric_transform(img, mapping_function,
                                output_shape=[720, 200], order=order)
    if ret_mapping:
        return polar, mapping_function
    return polar


def select_equal_paths(paths_a, paths_b):
    common = []
    i = 0
    for path in paths_a:
        for path2 in paths_b:
            if np.all(path == path2):
                common.append(i)
                break
        i += 1

    return common


def select_closed_curves(paths_a):
    closed = []
    for path in paths_a:
        if True:
            closed.append(path)
    return


def construct_distance_matrix(grad, seed):
    curr_start = seed[1]
    curr_end = seed[1]+1
    matrix = np.ones(grad.shape)*np.inf
    matrix[tuple(seed)] = 0

    for row in range(seed[0]-1, -1, -1):
        curr_start = max(curr_start-1, 0)
        curr_end = min(curr_end+1, grad.shape[1])

        for col in range(curr_start, curr_end):

            y1 = max(col-1, 0)
            y2 = min(col+2, grad.shape[1])
            negb_pos = np.meshgrid(row+1, np.arange(y1, y2))
            negb_pos = np.array(negb_pos).T.reshape([-1, 2])

            negb_dists = matrix[negb_pos[:, 0], negb_pos[:, 1]]
            negb_grads = grad[negb_pos[:, 0], negb_pos[:, 1]]

            pos = [row, col]
            geom_dists = np.sqrt(np.sum((negb_pos-pos)**2, axis=1))

            min_gradient = np.minimum(grad[tuple(pos)], negb_grads)
            grad_dists = alpha*np.exp(beta*(MAX_GRADIENT-min_gradient)) + delta

            dists = negb_dists + geom_dists * grad_dists
            matrix[row, col] = min(dists)

    return matrix


def find_all_paths(matrix, grad, seed):

    paths = []
    for i in range(0, seed[0]):
        j = np.argmin(matrix[i, :])
        final_point = (i, j)

        curr_point = final_point
        path = [final_point]
        for ii in range(i+1, seed[0]):

            (i, j) = curr_point
            y1 = max(j-1, 0)
            y2 = min(j+2, grad.shape[1])
            negb_pos = np.meshgrid(i+1, np.arange(y1, y2))
            negb_pos = np.array(negb_pos).T.reshape([-1, 2])

            negb_dists = matrix[negb_pos[:, 0], negb_pos[:, 1]]
            negb_grads = grad[negb_pos[:, 0], negb_pos[:, 1]]

            geom_dists = np.sqrt(np.sum((negb_pos-curr_point)**2, axis=1))

            min_gradient = np.minimum(grad[tuple(curr_point)], negb_grads)
            grad_dists = alpha*np.exp(beta*(MAX_GRADIENT-min_gradient)) + delta

            dists = negb_dists + geom_dists * grad_dists
            curr_point = negb_pos[np.argmin(dists)]
            path.append(curr_point)
        paths.append(path)
    return paths


def select_path(possible_paths, g_mag, th, n):
    for path in possible_paths:
        g_path = grad_seq(g_mag, path)
        counter = 0
        for i in g_path:
            counter = counter + 1 if i < th else 0
            if counter > n:
                break
        if counter <= n:
            return path

    # In case no path is found return the first one
    return possible_paths[-1]


def grad_seq(grad, path):
    seq = []
    for point in path:
        seq.append(grad[tuple(point)])
    return seq