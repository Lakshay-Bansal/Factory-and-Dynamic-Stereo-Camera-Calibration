import cv2
import numpy as np

def calibrate_fisheye(all_image_points, all_true_points, image_size, K, D, FLAGS, verbose=0):
#    print(all_true_points.shape, type(all_true_points))
    while True:
        assert len(all_true_points) > 0, "There are no valid images from which to calibrate."
        try:
            rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=all_true_points,
                imagePoints=all_image_points,
                image_size=image_size,
                K=K,
                D=D,
                flags=FLAGS,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            )
#            print('\t\tFound a calibration based on {} well-conditioned images.'.format(len(all_true_points)))
            return rms, mtx, dist, rvecs, tvecs
        except cv2.error as err:
            if(verbose==1):print(err)
            try:
                idx = int(str(err).split()[-4])  # Parse index of invalid image from error message
#                print(idx)
                all_true_points = np.delete(all_true_points,idx, 0)
                all_image_points = np.delete(all_image_points,idx, 0)
            except:
                raise err

def tprint():
    print("test print function!")
    print(cv2.__version__)