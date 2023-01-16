# Dynamic Stereo Camera Calibration

For 3D measurements and several other applications monocular and multi-camera calibration is an essential step. Both intrinsic and extrinsic parameters needs to be calculated for this. Typical solution is by using calibration patterns. But there are situations where the camera parameters can change following the initial calibration or situations where using such patterns is not feasible. Dynamic/self-calibration is the solution for such cases, wherein images of the 3D world are captured and using multiple such images and feature matching, the calibration is done. This project explores a solution for monocular and stereo camera calibration using multiple images captured in a given scene. It is assumed that the intrinsic parameters are known.


# Inroduction

Camera Calibration refers to the estimation of intrinsic camera parameters like focal length (principal point), the skew of a CCD array, and extrinsic camera parameters
which account for the relative position of a camera with respect to a world coordinate system. This needs a calibration pattern (e.g. checkerboard pattern). On the other
hand, dynamic camera calibration is carried out with real 3D scene images with dense features at varying depths. Nowadays there is a trend of a multi-camera system in
consumer devices with functionality like portrait mode (depth-based background blur in an image see Fig. 1.1), applications like an estimation of a dimension of an object
and its depth as shown in Fig. 1.2.

![depthBlur](https://user-images.githubusercontent.com/84389082/212613466-abcb8bf3-4f70-40ad-aa8a-327a825c251d.jpg | Hello World)


