#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:47:28 2023

@author: lakshayb
"""

import numpy as np
from scipy.optimize import least_squares
import cv2

def get_intrinsics(vals):
    # just optimise f
    f = vals[0]
    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = vals[1]
    K[1, 2] = vals[2]
    return K


#---------------------------Single Capture BA -----------------------------------------#
def singleCapture_reprojection_loss_function(opt_variables, features, num_pts):
    '''
    opt_variables --->  Camera Projection matrix + All 3D points
    '''

    K = get_intrinsics( opt_variables[0:3] )
    
    R_s_angles = opt_variables[3:6]
    R_s = cv2.Rodrigues(R_s_angles)[0]
    
    # C_s = opt_variables[6:9].reshape((3,1))
    # T_s = -1*np.matmul(R_s, C_s)
    
    # K = opt_variables[0:9].reshape(3,3)    
    # R_s = opt_variables[9:18].reshape(3,3) 
    T_s = opt_variables[6:9].reshape(3,1) 
    

    point_3d = opt_variables[9:].reshape((num_pts, 3))
    
    # Now append the array of residual that will used in the bundle adjustment
    rep_error = []
    # print(np.array(rep_error).shape)
    
    P_L = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    P_L = np.matmul(K, P_L)

    P_R = np.hstack([R_s, T_s])
    P_R = np.matmul(K, P_R)
    
    # For all left view image
    for idx, pt_3d in enumerate(point_3d):
        pt_2d = features[0][idx]
        reprojected_pt = np.matmul(P_L, np.hstack([pt_3d, 1]))
        reprojected_pt /= reprojected_pt[2]
        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append((pt_2d - reprojected_pt[0:2]))
    
    # For all right view image
    for idx, pt_3d in enumerate(point_3d): 
        pt_2d = features[1][idx]
        reprojected_pt = np.matmul(P_R, np.hstack([pt_3d, 1]))
        reprojected_pt /= reprojected_pt[2]
        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append((pt_2d - reprojected_pt[0:2]))
    
    # print(np.array(rep_error).shape)
    return np.array(rep_error).ravel()



def singleCapture_bundle_adjustment(features, K, R_s, T_s, points_3d):

    opt_variables = np.hstack((K.ravel(), R_s.ravel(), T_s.ravel(), points_3d.ravel() ))
    num_points = len(points_3d)
    
    corrected_values = least_squares(singleCapture_reprojection_loss_function, 
                                      opt_variables, 
                                      args=(features, num_points), 
                                      verbose=2, x_scale='jac', ftol=1e-2, xtol=1e-3, method='lm')
    
    # corrected_values = least_squares(singleCapture_reprojection_loss_function, 
    #                                   opt_variables, 
    #                                   args=(features, num_points), 
    #                                   verbose=2, ftol=1e-1, xtol=1e-3, method='trf')
    
    # print("The optimized values \n" + str(corrected_values))
    K_BA = get_intrinsics( corrected_values.x[0:3].reshape(3,1) )
    R_s_BA_angles = corrected_values.x[3:6].reshape(3,1)
    R_s_BA = cv2.Rodrigues(R_s_BA_angles)[0]
    
    T_s_BA = corrected_values.x[6:9].reshape(3, 1)
    
    points_3d_BA = corrected_values.x[9:].reshape((num_points, 3))

    return K_BA, R_s_BA, T_s_BA, points_3d_BA



#--------------------------------------------------------------------------------------#



#---------------------------Multi Capture BA -----------------------------------------#

# In this the triangulated 3D points during each view would be used as unknown during BA process
# And 3D points would get updated for there particular view only while K, R_s and T_s are the 
# joint parameter in all views
def multiCapture_reprojection_loss_function(opt_variables, features_all_view, keypoints, num_views):
    '''
    opt_variables --->  Camera Projection matrix + All 3D points
    '''

    K = get_intrinsics( opt_variables[0:3] )
    
    R_s_angles = opt_variables[3:6]
    R_s = cv2.Rodrigues(R_s_angles)[0]
    
    # C_s = opt_variables[6:9].reshape((3,1))
    # T_s = -1*np.matmul(R_s, C_s)
    
    # K = opt_variables[0:9].reshape(3,3)    
    # R_s = opt_variables[9:18].reshape(3,3) 
    T_s = opt_variables[6:9].reshape(3,1) 
    
    point_3d_all_view = []
    pos_start = 9
    pos_end = pos_start
    for npoints in keypoints:
        pos_end += npoints*3
        point_3d_all_view.append(opt_variables[pos_start:pos_end].reshape((npoints, 3)))
        
        pos_start = pos_end
                                 
    
    # Now append the array of residual that will used in the bundle adjustment
    rep_error = []
    # print(np.array(rep_error).shape)
    
    P_L = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    P_L = np.matmul(K, P_L)

    P_R = np.hstack([R_s, T_s])
    P_R = np.matmul(K, P_R)
    
    
    for (features, point_3d) in zip(features_all_view, point_3d_all_view):
        # For all left view image
        for idx, pt_3d in enumerate(point_3d):
            pt_2d = features[0][idx]
            reprojected_pt = np.matmul(P_L, np.hstack([pt_3d, 1]))
            reprojected_pt /= reprojected_pt[2]
            # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append((pt_2d - reprojected_pt[0:2]))
        
        # For all right view image
        for idx, pt_3d in enumerate(point_3d): 
            pt_2d = features[1][idx]
            reprojected_pt = np.matmul(P_R, np.hstack([pt_3d, 1]))
            reprojected_pt /= reprojected_pt[2]
            # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append((pt_2d - reprojected_pt[0:2]))
    
    # print(np.array(rep_error).shape)
    return np.array(rep_error).ravel()



def multiCapture_bundle_adjustment(features_all_view, K, R_s, T_s, points_3d_all_view, keypoints, num_views):

    opt_variables = np.hstack((K.ravel(), R_s.ravel(), T_s.ravel(), np.array(points_3d_all_view) ))
    
    # corrected_values = least_squares(multiCapture_reprojection_loss_function, 
    #                                   opt_variables, 
    #                                   args=(features_all_view, keypoints, num_views), 
    #                                   verbose=2, x_scale='jac', ftol=1e-2, xtol=1e-3, method='lm')
    
    corrected_values = least_squares(multiCapture_reprojection_loss_function, 
                                      opt_variables, 
                                      args=(features_all_view, keypoints, num_views), 
                                      verbose=2, x_scale='jac', ftol=1e-1, xtol=1e-3, method='trf')
    
    # print("The optimized values \n" + str(corrected_values))
    K_BA = get_intrinsics( corrected_values.x[0:3].reshape(3,1) )
    R_s_BA_angles = corrected_values.x[3:6].reshape(3,1)
    R_s_BA = cv2.Rodrigues(R_s_BA_angles)[0]
    
    T_s_BA = corrected_values.x[6:9].reshape(3, 1)
    
    points_3d_BA = []
    pos_start = 9
    pos_end = pos_start
    for npoints in keypoints:
        pos_end += npoints*3
        points_3d_BA.append(corrected_values.x[pos_start:pos_end].reshape((npoints, 3)))
        
        pos_start = pos_end

    return K_BA, R_s_BA, T_s_BA, points_3d_BA

#--------------------------------------------------------------------------------------#