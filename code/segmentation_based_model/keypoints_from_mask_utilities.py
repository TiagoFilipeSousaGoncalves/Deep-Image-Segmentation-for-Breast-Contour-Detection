# Imports 
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt

# Segmentation Based Model Functions
# Euclidean Distance to Check Points Distances
def compute_euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

# Keypoints from Masks Function
def get_keypoints_from_breasts_masks(breasts_masks, threshold_kpts=400, PLOTS=False):
    contours_list = []
    n = int(breasts_masks.shape[0])
    
    for i in range (n):
        # gray_image = masks[i].reshape((384, 512))
        # gray_image *= 255
        gray_image = breasts_masks[i]

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(gray_image, 1.0)
        contours_list.append(contours)

        if PLOTS:
            # Display the image and plot all contours found
            fig, ax = plt.subplots()
            ax.imshow(gray_image, cmap='gray')

            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

                ax.axis('image')
                # ax.set_xticks([])
                # ax.set_yticks([])
                plt.show()

    # Manual Threshold to Detect Breast's Contours
    # Split contours by keypoints size
    # Iterate over the contour list obtained previously
    for i in range(np.shape(contours_list)[0]):
        # print(np.shape(contour))
        if np.shape(contours_list[i])[0] > 1:
            # print("Index {}:".format(i))
            counter = 0
            indexes = []
            # print(np.shape(contours_list[i]))
            for j in range(np.shape(contours_list[i])[0]):
                if np.shape(contours_list[i][j])[0] > threshold_kpts:
                    counter += 1
                    indexes.append(j)
                    # print(np.shape(contours_list[i][j]))
                    # print(j)
                
    
            if counter == 1:
                contours_list[i] = contours_list[i][indexes[0]]
            
            elif counter == 2:
                tmp = np.concatenate((contours_list[i][indexes[0]], contours_list[i][indexes[1]]))
                contours_list[i] = tmp

    # Reshape Contours
    for i in range(np.shape(contours_list)[0]):
        if len(np.shape(contours_list[i])) > 2:
            # print(np.shape(contours_list[i]))
            tmp = np.array(contours_list[i]).reshape((-1, 2))
            # print(tmp.shape)
            contours_list[i] = tmp

    
    return contours_list


# Plot Keypoints from Breast Masks Contours Function
def plot_keypoints_from_breast_masks_contours(breasts_masks, contours_keypoints):
    # Let's Plot Breast's Contours
    n = int(breasts_masks.shape[0])
    for i in range(n):
        # gray_image = masks[i].reshape((384, 512))
        # gray_image *= 255
        gray_image = breasts_masks[i]

        # plt.plot(hybrid_contours_list[i][:,1], hybrid_contours_list[i][:, 0], 'o')
        plt.axis('off')
        plt.plot(contours_keypoints[i][:, 1], contours_keypoints[i][:, 0], 'o')
        plt.imshow(gray_image, cmap='gray')
        plt.savefig(str(i), bbox_inches='tight', pad_inches=0.0)
        plt.show()


# Function to perform Segmentation Based Model Predictions
def mix_isbi_and_contours(contours_keypoints, isbi_preds):
    # Convert processed preds to column array notation
    isbi_preds = np.array(isbi_preds)
    isbi_preds *= 512
    processed_preds = isbi_preds.copy()
    
    preds_column_array = []

    for i in range(processed_preds.shape[0]):
        tmp_list = []
        for j in range(processed_preds[i].shape[0]):
            if j % 2 == 0:
                tmp_list.append([isbi_preds[i][j+1], isbi_preds[i][j]])
  
        preds_column_array.append(np.array(tmp_list))

    preds_column_array = np.array(preds_column_array)
    
    # print(np.shape(preds_column_array))
    # print(np.shape(contours_keypoints))

    for index in range(np.shape(preds_column_array)[0]):
        for i in range(np.shape(preds_column_array)[1]-2):
            distances = []
            for j in range(np.shape(contours_keypoints[index])[0]):
                distances.append(compute_euclidean_distance(preds_column_array[index][i], contours_keypoints[index][j]))
      
            # print(np.argmin(distances))
            if len(distances) > 0:
                if contours_keypoints[index][np.argmin(distances)] in preds_column_array[index]:
                    preds_column_array[index][i] = preds_column_array[index][i]
                else:
                    try:
                        preds_column_array[index][i] = contours_keypoints[index][np.argmin(distances)]
                    except:
                        preds_column_array[index][i] = preds_column_array[index][i]

    # print(np.shape(preds_column_array))
    
    return preds_column_array


# Convert predictions to the ISBI notation
def mixed_to_our_notation(preds_column_array):
    # Let's Convert Column Notation To Our Notation
    # x_coord are even numbers, y_coord are odd numbers

    final_contours = []

    for i in range(np.shape(preds_column_array)[0]):
        tmp = []
        for array in preds_column_array[i]:
            tmp.append(array[1])
            tmp.append(array[0])
  
        # print(np.shape(tmp))
        final_contours.append(np.array(tmp))

    print(np.shape(final_contours))

    return np.array(final_contours)