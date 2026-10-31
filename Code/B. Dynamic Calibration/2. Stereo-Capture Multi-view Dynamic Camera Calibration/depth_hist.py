#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:42:33 2023

@author: lakshayb
"""
# For all views of stereo pair

import cv2
import numpy as np
import matplotlib.pyplot as plt


def depth(imgL, imgR, view, K, D_left, D_right, R_s, T_s):
    
    ## Convert BGR images to RGB
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
     
    plt.imshow(np.hstack([imgL, imgR]), 'gray') 
    plt.title(f"View={view} --- Left and Right Images")
    plt.show()

    # # creates StereoBm object 
    # stereo = cv2.StereoSGBM.create(minDisparity = 1, numDisparities = 64,
    #                             P1 = 20, P2 = 200, uniquenessRatio = 5 )
      
    # # computes disparity for unrectified images which is not required
    # disparity = stereo.compute(imgL, imgR)
      
    # # displays image as grayscale and plotted
    # plt.imshow(disparity, 'gray')
    # plt.title("{} - disparity map by {} method ".format(fileName, "SGBM"))
    # plt.show()

    # baseline = T_s[0][0]
    # focal_length = K[0][0]  # 100 pixels
    # depth = (baseline * focal_length) / disparity

    # plt.imshow(depth, 'gray')
    # plt.title("'Depth Map by un-rectified images")
    # plt.show()

    # Now rectify the images and then estimating disparity followed by depth map
    # Compute the rectification transforms for each camera
    
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K, D_left, K, D_right, imgL.shape[:2], R_s, T_s)

    # Compute the map matrices for each camera
    # While passing imgL.shape[:2] (which return rows, cols dimension of image) as 
    # the size of undistored image on result should be of size (cols, rows) hence we need to reverse the size
    # Keep in mind to change the order of size 
    map1_left, map2_left = cv2.initUndistortRectifyMap(K, D_left, R1, P1, [imgL.shape[:2][1], imgL.shape[:2][0]], cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K, D_right, R2, P2, [imgR.shape[:2][1], imgR.shape[:2][0]], cv2.CV_16SC2)

    # Rectify the images using the map matrices
    img_left_rect = cv2.remap(imgL, map1_left, map2_left, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(imgR, map1_right, map2_right, cv2.INTER_LINEAR)

    plt.imshow(np.hstack([img_left_rect, img_right_rect]), 'gray')
    plt.title(f"View={view} --- Rectified Images")
    plt.show()


    # Compute the disparity map using the StereoSGBM algorithm, now on Rectified Images
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7)
    disparity_rect = stereo.compute(img_left_rect, img_right_rect)

    plt.imshow(disparity_rect, 'gray')
    plt.title(f"View={view} --- disparity map on Rectified Images")
    plt.show()

    # Compute the depth map using the known baseline distance and focal length
    # baseline = 0.052  # 5.2 cm it takes values in meter
    baseline = T_s[0][0]
    focal_length = K[0][0]  # 100 pixels
    depth_rect = (baseline * focal_length) / disparity_rect
    
    # # Set the negative and infinite values to NaN
    # depth_rect[depth_rect < 0] = np.nan
    # depth_rect[np.isinf(depth_rect)] = np.nan
    
    # # Normalize the depth map between 0 and 1
    # depth_rect_norm = 255 * (depth_rect - np.nanmin(depth_rect)) / (np.nanmax(depth_rect) - np.nanmin(depth_rect))

    plt.imshow(depth_rect, 'gray')
    plt.title(f"View={view} --- Depth Map from rectified images")
    plt.show()
    
    return depth_rect


def histogram(depth, view):
    # Define the bin size and depth ranges
    b = 5
    d = 3
    # depth_ranges = [(0, 4), (4, 8), (8, np.inf)]
    depth_ranges = [(0, 4), (4, 8), (8, 12)]

    # Compute the z-depth of each feature point using single capture dynamic calibration
    # depth variable
    # depth = depth_rect

    # Initialize the bin counts for each bin
    bin_counts = np.zeros((b, b, d))

    # We are finding the number of pixel in image that correspond to ith and jth bin
    width_block_img = int(depth.shape[0] / b)
    height_block_img = int(depth.shape[1] / b)


    # Assign each feature point to a bin based on its pixel location and z-depth
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            # Compute the bin indices for the pixel location
            bin_i = int(i / width_block_img) 
            bin_j = int(j / height_block_img)
            
            # Find the depth range for the z-depth of the feature point
            bin_depth = -1
            for k, (depth_min, depth_max) in enumerate(depth_ranges):
                if depth[i, j] >= depth_min and depth[i, j] <= depth_max:
                    bin_depth = k
                    break
            
            # Increment the count for the corresponding bin
            if bin_depth != -1:
                bin_counts[bin_i, bin_j, bin_depth] += 1

    # Flatten the bin counts into a 1D feature vector
    feature_vector = bin_counts.flatten()


    # Plot the barplot
    plt.bar(list(range(len(feature_vector))) , feature_vector, width = 0.5)
    plt.title(f'Histogram of feature at different depth from rect img view={view}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    return feature_vector


def depth_hist(imgL, imgR, view, K, D_left, D_right, R_s, T_s):
    depth_map = depth(imgL, imgR, view, K, D_left, D_right, R_s, T_s)
    return histogram(depth_map, view)

# # Read dataset
# from glob import glob

# camera_ip = ["192.168.1.72", "192.168.1.69"]
# trans = str("combine")

# imagesL = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/Realscene/{trans}/image_*_{camera_ip[0]}.jpg")
# imagesR = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/Realscene/{trans}/image_*_{camera_ip[1]}.jpg")

# imagesL.sort()
# imagesR.sort()

# # Camera calibration parameter from checkerboard pattern inbuilt calibration

# K =  np.array([[764.45896419,   0.  ,       952.31442344],
#                 [  0.,         762.55114182, 613.31775089],
#                 [  0.,           0.,         1.        ]])

# R_s = np.array([[ 9.97953666e-01, -1.38733611e-03, -6.39261742e-02],
#                 [ 4.37874904e-04,  9.99889428e-01, -1.48640812e-02],
#                 [ 6.39397272e-02,  1.48056726e-02,  9.97843927e-01]])

# T_s = np.array([[ 3.84821196],
#                 [-0.32734609],
#                 [-0.12060897]])

# D_left = np.array([[ 0.02385932],
#         [ 0.14622246],
#         [-0.14332024],
#         [ 0.04507821]])

# D_right = np.array([[ 0.02385932],
#         [ 0.14622246],
#         [-0.14332024],
#         [ 0.04507821]])

# H_i = []
# D = []
    
# def main():
#     for view, (imgLPath, imgRPath) in enumerate(zip(imagesL, imagesR)):
#         # Depth and then histogram
#         imgL = cv2.imread(imgLPath)
#         imgR = cv2.imread(imgRPath)
#         depth_map = depth(imgL, imgR, view, K, D_left, D_right, R_s, T_s)
#         D.append(depth_map)
#         H_i.append(histogram(depth_map, view))
#         break
        

# if __name__ == "__main__":
#     main()