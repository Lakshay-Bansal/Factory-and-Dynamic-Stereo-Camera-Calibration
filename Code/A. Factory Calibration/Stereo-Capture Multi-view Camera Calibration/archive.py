# For debuging purpose
## Plotting single detected corner and reproject point after BA in image

# for pos, view in enumerate(final_view):
    
#     # P_l = P_l_afterBA[view]
#     # P_r = P_r_afterBA[view]
    
#     # P_l = projectionMatrices_L_before[view]
    
#     r_L_matrix , _ = cv2.Rodrigues( P_l_afterBA[pos][0] )
#     C_L = P_l_afterBA[pos][1].reshape((3, 1))
#     P = np.hstack([r_L_matrix, -1*np.matmul(r_L_matrix, C_L)])
    
#     dummy = np.hstack([R_L_after_BA, np.zeros((3,1)) ])
#     dummy2 = np.vstack([P, [0, 0, 0, 1]])
#     P_l = np.matmul(dummy, dummy2)
    
#     # dummy = np.hstack([R_after_BA, T_after_BA])
#     # dummy2 = np.vstack([P_l, [0, 0, 0, 1]])
    
#     # P_r = np.matmul(dummy, dummy2)
    
#     # P_LR = [P_l, P_r]
#     # points_2d_LR = [pts_2d_left[view], pts_2d_right[view]]
    
#     # for P, points_2d in zip(P_LR, points_2d_LR):
#     #     # print(points_2d.shape)
#     error = 0
#     for idx, pt_3d in enumerate(common_3D_pts_after_BA):
#         # print(pt_3d.shape)
#         # Kth view, xth camera and jth matched feature 
#         pt_2d = np.array([pts_2d_left[view-1][idx][0], pts_2d_left[view-1][idx][1]])
#         reprojected_pt = np.float32(np.matmul(np.matmul(K, P_l), pt_3d))
#         reprojected_pt /= reprojected_pt[2]
        
#         error += np.sum((pt_2d - reprojected_pt[0:2])**2)
#         # print(f"{idx} - {error}")

#         # Ploting Image
#         # path_idx = left_img_idx[view+1]
#         # path = imagesL[path_idx]
#         path = imagesL[view]
#         image = cv2.imread(path)
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