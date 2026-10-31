import numpy as np

# Selecting only good views - threshold value of thresh_val on per point rpe in an image per view

def threshold_view_selection(projectionMatrices_L_before, projectionMatrices_R_before, pts_2d_left, pts_2d_right,
                             imagesL, imagesR, common_3D_pts, K, left_img_idx, threshold = 900):
    
    rpe_point_view = [] 
    final_view = []
    threshold = threshold

    for pos in range(len(projectionMatrices_L_before)):
        
        P_l = projectionMatrices_L_before[pos]
        P_r = projectionMatrices_R_before[pos]
        
        P_lr = [P_l, P_r]
        # Since Iref = 0th instance of left_img_idx, for that P_l and P_r are not calculated
        # corner_2D_lr = [imgpointsL[view+1][0], imgpointsR[view+1][0]]   #See view =0 is reference frame while P_l estimation
        corner_2D_lr = [pts_2d_left[pos], pts_2d_right[pos] ]
        path_lr = [imagesL, imagesR]
        rev_idx = 0
        
        a=[]  
        for P, corner_2D in zip(P_lr, corner_2D_lr):
            # # #To view detected corner and reprojected points for debugging purpose
            # path = ''
            # tag =''
            # if rev_idx == 0:
            #     path = path_lr[0][left_img_idx[pos+1]] 
            #     Image view = left_img_idx[pos+1]
            #     # As left_img_idx[0] is the reference frame
            #     # for which there is no P and 2D points for it are stored
            #     tag = 'Left'

            # else:
            #     path = path_lr[1][left_img_idx[pos+1]]
            #     tag = 'Right'
                
            for idx, pt_3d in enumerate(common_3D_pts):
                reprojected_pt = np.float32(np.matmul(np.matmul(K, P), pt_3d))
                reprojected_pt /= reprojected_pt[2]
                
                rpe_point =  np.array(corner_2D[idx] - reprojected_pt[0:2]).ravel()
                # print(idx, rpe_point)
                # print(left_img_idx[pos+1], rpe_point, end="\n\n")
                a.extend(rpe_point)
                
            #     # #To view detected corner and reprojected points for debugging purpose
            #     # print(path)
            #     image = cv2.imread(path)
            #     plt.imshow(image)
            #     plt.scatter(corner_2D[idx][0], corner_2D[idx][1], color='blue')
            #     plt.scatter(reprojected_pt[0], reprojected_pt[1], color='red')
            #     plt.title(f"{tag} View - {left_img_idx[pos+1]} and Point - {idx} - Point - {reprojected_pt[:2]}")
            #     plt.show()
            # rev_idx = ~rev_idx
            
        print(max(abs(np.array(a))))
        rpe_point_view.append(max(abs(np.array(a))))
        
        if max(abs(np.array(a))) < threshold:
            final_view.append(left_img_idx[pos+1])

    print(final_view)
    return final_view