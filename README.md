# Dynamic Stereo Camera Calibration

For 3D measurements and several other applications monocular and multi-camera calibration is an essential step. Both intrinsic and extrinsic parameters needs to be calculated for this. Typical solution is by using calibration patterns. But there are situations where the camera parameters can change following the initial calibration or situations where using such patterns is not feasible. Dynamic/self-calibration is the solution for such cases, wherein images of the 3D world are captured and using multiple such images and feature matching, the calibration is done. This project explores a solution for monocular and stereo camera calibration using multiple images captured in a given scene. It is assumed that the intrinsic parameters are known.


# Inroduction

Camera Calibration refers to the estimation of intrinsic camera parameters like focal length (principal point), the skew of a CCD array, and extrinsic camera parameters
which account for the relative position of a camera with respect to a world coordinate system. This needs a calibration pattern (e.g. checkerboard pattern). On the other
hand, dynamic camera calibration is carried out with real 3D scene images with dense features at varying depths. Nowadays there is a trend of a multi-camera system in
consumer devices with functionality like portrait mode (depth-based background blur in an image see Fig. 1:[Portrait mode](https://user-images.githubusercontent.com/84389082/212613466-abcb8bf3-4f70-40ad-aa8a-327a825c251d.jpg), applications like an estimation of a dimension of an object
and its depth as shown in Fig. 2:[Object dimension detection in an image](https://user-images.githubusercontent.com/84389082/212614651-b8a30d7f-806f-43e8-be54-a028f54bac56.png).

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212613466-abcb8bf3-4f70-40ad-aa8a-327a825c251d.jpg" /><br>
  <b>Fig. 1:  Portrait mode</b>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212614651-b8a30d7f-806f-43e8-be54-a028f54bac56.png" /><br>
  <b>Fig. 2:  Object dimension detection in an image</b>
</p>


An accurately calibrated camera is needed for the above-mentioned applications to work. Factory calibration (when the camera is delivered from the manufacturing
plant) is not valid because parameters change over the operating lifetime of the device due to normal wear and tear, and thermal effects (heat generation during the operation of the device).

Possible causes of change in camera parameters over the lifetime of camera operation are: 

a) Thermal heat generated from the camera during its operation can cause the focal length of the lens or CCD array expansion which doesn’t match with the factory calibration ([Fig. 3a](https://user-images.githubusercontent.com/84389082/212615261-6871efb1-c53d-4e7e-bf15-cf19436f6864.jpg)),

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212615261-6871efb1-c53d-4e7e-bf15-cf19436f6864.jpg" /><br>
  <b>Fig. 3a: Effect of thermal heat on camera geometry</b>
</p>


b) Because of mechanical stress the printed circuit board (PCB) attached to the camera module can bend which alters the camera pose see [Fig. 3b](https://user-images.githubusercontent.com/84389082/212615265-23d399bf-0f68-4ab3-a3de-74e8d2832679.jpg),

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212615265-23d399bf-0f68-4ab3-a3de-74e8d2832679.jpg" /><br>
  <b>Fig. 3b: Change in camera pose owing to mechanical stress</b>
</p>


c) The non-rigid camera component can move and will change the calibration parameters as shown in [Fig. 1.3c](https://user-images.githubusercontent.com/84389082/212615267-c678d5a4-2e7b-4c23-be8c-6765225aa66d.jpg). Camera calibration at the consumer end is not feasible due to the requirement of buying accurate calibration patterns and there after collecting calibration data. Hence dynamic camera calibration is an alternate economical and scalable way of camera calibration.

<p align="center">
  <img src="https://user-images.githubusercontent.com/84389082/212615267-c678d5a4-2e7b-4c23-be8c-6765225aa66d.jpg" /><br>
  <b>Fig. 3c: Change in camera parameter due to non-rigid camera component</b>
</p>
