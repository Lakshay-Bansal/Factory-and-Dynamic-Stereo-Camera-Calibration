# Camera Model
The pinhole camera is as shown in Fig. 3.1. The light reflected from an object that needs to be captured passes through a pinhole and the image is formed on the screen of the camera (inverted w.r.t. orientation of the object in the real world i.e. captured by camera). In the pin-hole camera model, it is assumed that all light rays are passed through the pinhole’s center before being imaged on the screen. The pinhole camera model does not account for lens distortion because an ideal pinhole camera does not have a lens. 

To accurately represent a real camera, the full camera model used by the algorithm of camera calibration includes radial and
tangential lens distortion. Usually, in the full camera model, only radial distortion is considered as shown in Fig. 3.2.

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/5a6927f8-1733-4e57-9da0-e1a44b80f237.png" /><br>
  <b>Fig. 3.1: Pin hole camera model</b><br>
  Source: https://in.mathworks.com/help/vision/ug/camera-calibration.html
</p>

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/cf7d1d8a-192b-4af3-84bb-8298fbea9eed.png" /><br>
  <b>Fig. 3.2: Radial distortion due to lens</b><br>
  Source: https://in.mathworks.com/help/vision/ug/camera-calibration.html
</p>

As shown in Fig. 3.3, the point P (Xw, Yw, Zw) in world coordinate is transformed to (Xc, Yc, Zc) from the reference frame of camera coordinate system by applying rotation and translation to the world point. Which is then mapped to image plane by perspective transformation that operation is captured by the camera intrinsic matrix.

<p align="center">
  <img src="https://github.com/Lakshay-Bansal/Dynamic-Stereo-Camera-Calibration/assets/84389082/d385b436-645f-44b4-9251-7678fe2b33ac.png" /><br>
  <b>Fig. 3.3: Image formation of pin hole camera via a perspective transform</b><br>
</p>

