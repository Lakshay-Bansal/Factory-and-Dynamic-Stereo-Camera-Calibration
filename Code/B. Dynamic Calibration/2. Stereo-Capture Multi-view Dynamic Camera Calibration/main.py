#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:15:54 2023

@author: lakshayb
"""

import cv2
import numpy as np
from glob import glob
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from useful_fn import plot_Matchfeature, plot_3D_points, reprojection_3D_pt, rep_error_fn


# Bundle adjustment functions__________________________________________________
from bundle_adjustment import singleCapture_bundle_adjustment, multiCapture_bundle_adjustment

#----------------------------------------------------------------------------------------------#


# Depth and Histogram__________________________________________________
from depth_hist import depth_hist

#----------------------------------------------------------------------------------------------#


#------------------Optimality Criteria-------------------------------------------#
def stddev(histogram):
    # Find the indices of non-zero entries in both arrays
    indices = np.nonzero(histogram)
    std_dev = np.std(indices)
    return std_dev


def optimality_criteria(keypoints, E_I, h_S, h_I, alpha=0.15, beta=0.5, gamma=0.35):
    
    I_opt = alpha * np.array(keypoints) + \
            beta * (1/np.array(E_I)) + \
            gamma * np.array([stddev(h) for h in np.array(h_S) + np.array(h_I)])
    
    return np.argmax(I_opt)
#----------------------------------------------------------------------------------#



# Camera calibration parameter from checkerboard pattern inbuilt calibration
K =  np.array([[764.45896419,   0.  ,       952.31442344],
               [  0.,         762.55114182, 613.31775089],
               [  0.,           0.,         1.        ]])

R_s = np.array([[ 9.97953666e-01, -1.38733611e-03, -6.39261742e-02],
                [ 4.37874904e-04,  9.99889428e-01, -1.48640812e-02],
                [ 6.39397272e-02,  1.48056726e-02,  9.97843927e-01]])

T_s = np.array([[ 3.84821196],
                [-0.32734609],
                [-0.12060897]])


D_left = np.array([[ 0.02385932],
        [ 0.14622246],
        [-0.14332024],
        [ 0.04507821]])

D_right = np.array([[ 0.02385932],
        [ 0.14622246],
        [-0.14332024],
        [ 0.04507821]])

#----------------------------------------------------


# Read dataset

camera_ip = ["192.168.1.72", "192.168.1.69"]
trans = str("combine_D1+D2")

imagesL = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/Realscene/{trans}/image_*_{camera_ip[0]}.jpg")
imagesR = glob(rf"/mnt/Data2/Datatset/Apr23_StereoDataset/Realscene/{trans}/image_*_{camera_ip[1]}.jpg")

imagesL.sort()
imagesR.sort()

imagesL = imagesL[:10]
imagesR = imagesR[:10]

# Create SIFT object 
sift = cv2.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)

P_L = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
P_L = np.matmul(K, P_L)

P_R = np.hstack([R_s, T_s])
P_R = np.matmul(K, P_R)

# Global Variables -------------------------------------------------------------

# Reprojection Error for final viewsrep_error_fn(P, features, pts_3D)
rpe = {'beforeBA': [], 'afterBA': []}

points_3d_all_view = []
features_all_view = []

# store the histogram feature vector of size 5*5*3 = 75 for each pair of stereo images 
# Is being calculated from the depth map that is obatined from stereo images of each view
h_I = []

# store the number of keypoint in used in the particular view during BA
keypoints = []
    
for view in range(len(imagesL)):
    
    imgL = cv2.imread(imagesL[view])
    imgR = cv2.imread(imagesR[view])
    
    # Finding sift keypoints and descriptor
    kp_L, desc_L = sift.detectAndCompute(imgL,None)
    kp_R, desc_R = sift.detectAndCompute(imgR,None)

    
    matches = flann.knnMatch(desc_L, desc_R, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts1.append(kp_L[m.queryIdx].pt)
            pts2.append(kp_R[m.trainIdx].pt)
            
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Directly computing E matrix
    E, mask = cv2.findEssentialMat(pts1,
                              pts2, 
                              K,
                              method=cv2.RANSAC, 
                              prob=0.99,
                              threshold=0.75)
    
    retval, R_s_E, T_s_E, mask = cv2.recoverPose(E, pts1, pts2, K)

    # # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    
    # First optimality criteria --------------------------------
    keypoints.append(len(pts1))
    
    print(view)
    plot_Matchfeature(imagesL[view], pts1, 0, view)
    # Right
    plot_Matchfeature(imagesR[view], pts2, 1, view)

    points_3d = cv2.triangulatePoints(P_L, P_R, pts1.T, pts2.T)
    points_3d /= points_3d[-1]
    points_3d = points_3d.T
    
    points_3d_all_view.append(points_3d.T[:3].T)
    
    plot_3D_points(points_3d.T[:3].T, view)

    # Reprojeting 3D points
    reprojection_3D_pt(path = imagesL[view], 
                       features = pts1, flag=0, 
                       pos=view, 
                       pts_3D = points_3d, 
                       P = P_L)
    
    reprojection_3D_pt(path = imagesR[view], 
                       features = pts2, flag=1, 
                       pos=view, 
                       pts_3D = points_3d, 
                       P = P_R)

    # Re-projection error
    # rpe['beforeBA'].append(rep_error_fn(P_L, pts1, points_3d.T)) #--Left
    # rpe['beforeBA'].append(rep_error_fn(P_R, pts2, points_3d.T)) #--Right
    rpe_view = rep_error_fn(P_L, pts1, points_3d) + rep_error_fn(P_R, pts2, points_3d)
    
    # Second optimality criteria --------------------------------
    rpe['beforeBA'].append(rpe_view)
    
    ## Bundle Adjustment
    print("Performing bundle adjustemnt")
    features = [pts1, pts2]
    features_all_view.append(features)
    
    t0 = time.time()
    K_BA, R_s_BA, T_s_BA, points_3d_afterBA = singleCapture_bundle_adjustment(features, 
                                                                              np.array([ K[0, 0], K[0, 2], K[1, 2] ]), 
                                                                              cv2.Rodrigues(R_s)[0].ravel(), 
                                                                              T_s, 
                                                                              points_3d.T[:3].T )   
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    
    
    P_L_BA = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    P_L_BA = np.matmul(K_BA, P_L_BA)

    P_R_BA = np.hstack([R_s_BA, T_s_BA])
    P_R_BA = np.matmul(K_BA, P_R_BA)
    
    points_3d_afterBA = np.hstack([points_3d_afterBA, 
                        np.ones((len(points_3d_afterBA), 1)) ])
    
    # Reprojecting 3D points after BA
    reprojection_3D_pt(path = imagesR[view], 
                        features = pts2, flag=1, 
                        pos=view, 
                        pts_3D = points_3d_afterBA, 
                        P = P_R_BA)
    
    rpe_view_afterBA = rep_error_fn(P_L_BA, pts1, points_3d_afterBA) + rep_error_fn(P_R_BA, pts2, points_3d_afterBA)
    rpe['afterBA'].append(rpe_view_afterBA)
    

    # Third optimality criteria --------------------------------
    h_view = depth_hist(imgL, imgR, view, K, D_left, D_right, R_s, T_s)
    h_I.append(h_view)


# Storing the copy of variable which will get updated in Multi-capture algorithm
copy_keypoints = keypoints
copy_rpe = rpe
copy_h_I = h_I

#%%
# it will store the index of the frame seleted for final calibration after multi-capture calibration

# 0th index of images as a refrenece frame
# Initialising first frame as an reference frame
# I_opt will give the optimal views suitable for calibration by all_view - S
ref_idx = 0
S = [ref_idx]
rpe_Ref = rpe['afterBA'][ref_idx]
h_S = h_I[ref_idx]

result_dict = {"S": S, "RPE": rpe_Ref, "histogram": h_S, "camParameters": [R_s, T_s]}

### # Delete the corresponding data of view that are for ref_idx
### del rpe['afterBA'][ref_idx], h_I[ref_idx]


from bundle_adjustment import multiCapture_bundle_adjustment
# opt_view = optimal view
while (True):
    # Not considering the h_I for ref_idx position = 0
    I_opt = optimality_criteria(keypoints = keypoints[1:], 
                                E_I = rpe['afterBA'][1:], 
                                h_S = h_S, 
                                h_I = h_I[1:], 
                                alpha=0.15, beta=0.5, gamma=0.35)
    print("I_opt: ", I_opt)
    
    S.append(I_opt+1)
    
    points_3d_all_view_1D = []

    for p3d in [points_3d_all_view[v] for v in S]:
        points_3d_all_view_1D.extend(np.array(p3d).ravel())
    
    
    # In multi-capture BA the 'lm' method is giving error related to size of memory outbound for variable
    K_BA, R_s_BA, T_s_BA, points_3d_afterBA_all_view = multiCapture_bundle_adjustment([features_all_view[v] for v in S], 
                                                                                      K, R_s, T_s, 
                                                                                      points_3d_all_view_1D,
                                                                                      [keypoints[v] for v in S],
                                                                                      num_views = len(S) )
    
    P_L_BA = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    P_L_BA = np.matmul(K_BA, P_L_BA)

    P_R_BA = np.hstack([R_s_BA, T_s_BA])
    P_R_BA = np.matmul(K_BA, P_R_BA)
    
    # Not needed ---------------------------------------------------------------
    # rpe_S = 0
    # for (features, point_3d) in zip(features_all_view[S], point_3d_all_view[S]):          
    #     points_3d_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1)) ])
    
    #     rpe_view = rep_error_fn(P_L_BA, features[0], points_3d_homogeneous) + \
    #                        rep_error_fn(P_R_BA, features[1], points_3d_homogeneous)
        
    #     rpe_S += rpe_view
    #-----------------------------------------------------------------------------_
    
    # RPE of I_ref frame which is first frame in the dataset or at 0th index
    # by the K, R_s, T_s from the multiCapture_bundle_adjustment()
    points_3d_homogeneous = np.hstack([points_3d_afterBA_all_view[ref_idx], 
                                       np.ones((len(points_3d_afterBA_all_view[ref_idx]), 1)) 
                                       ])
    
    rpe_Ref_byS = rep_error_fn(P_L_BA, features_all_view[ref_idx][0], points_3d_homogeneous) + \
                       rep_error_fn(P_R_BA, features_all_view[ref_idx][1], points_3d_homogeneous)
    
    if rpe_Ref_byS < rpe_Ref:
        
        # Updating parameters
        rpe_Ref = rpe_Ref_byS
        h_S = np.sum(h_I[S], axis=0)
         
        result_dict["S"] = S, 
        result_dict["RPE"] = rpe_Ref
        result_dict["histogram"] = h_S
        result_dict["camParameters"] = [R_s_BA, T_s_BA]
        
        keypoints[I_opt] = 0
        rpe['afterBA'][I_opt] = np.inf
        h_I[I_opt] = [0] * len(h_I[0])
        
    else:
        S.pop()
        break

#%% Extra

# #Testing Purpose: reprojected points by P_L and P_R on trinagulated 3D points from stereo view
# # Plotting single detected corner and reproject point after BA in image


# view = 0
# error = [ ]
# for (features, point_3d) in zip(features_all_view, points_3d_all_view):
#     # For all left view image
#     path = imagesL[view]
#     image = cv2.imread(path)
#     for idx, pt_3d in enumerate(point_3d):
#         # print(pt_3d.shape)
#         pt_2d = features[0][idx]
#         reprojected_pt = np.matmul(P_L, np.hstack([pt_3d, 1]))
#         reprojected_pt /= reprojected_pt[2]
#         # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
#         error.append(sum((pt_2d - reprojected_pt[0:2])**2)**0.5)
#         print(idx, sum((pt_2d - reprojected_pt[0:2])**2)**0.5)
        
#         # Image
#         plt.imshow(image)
#         plt.scatter(pt_2d[0], pt_2d[1], color='red')
#         plt.scatter(reprojected_pt[0], reprojected_pt[1], color='blue')
#         plt.title(f"{idx} - Point - {reprojected_pt[:2]}")
#         # if flag ==0:
#         #     plt.title(f"{pos} Left Image - Detected Corners")
#         # elif flag==1:
#         #     plt.title(f"{pos} Right Image - Detected Corners")
#         plt.show()
    
#     path = imagesR[view]
#     image = cv2.imread(path)
#     # For all right view image
#     for idx, pt_3d in enumerate(point_3d): 
#         pt_2d = features[1][idx]
#         reprojected_pt = np.matmul(P_R, np.hstack([pt_3d, 1]))
#         reprojected_pt /= reprojected_pt[2]
#         # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
#         error.append((pt_2d - reprojected_pt[0:2]))
#         print(idx, sum((pt_2d - reprojected_pt[0:2])**2)**0.5)
        
#         # Image
#         plt.imshow(image)
#         plt.scatter(pt_2d[0], pt_2d[1], color='red')
#         plt.scatter(reprojected_pt[0], reprojected_pt[1], color='blue')
#         plt.title(f"{idx} - Point - {reprojected_pt[:2]}")
#         # if flag ==0:
#         #     plt.title(f"{pos} Left Image - Detected Corners")
#         # elif flag==1:
#         #     plt.title(f"{pos} Right Image - Detected Corners")
#         plt.show()

#     break
    