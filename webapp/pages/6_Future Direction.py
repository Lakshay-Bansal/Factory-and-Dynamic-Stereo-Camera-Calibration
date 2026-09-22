import streamlit as st
from pybtex.database.input import bibtex

st.title("Future Directions")
st.write('''
In this study, the calibration of the stereo camera was performed up to metric reconstruction, 
as the scale factor for the calibrated camera was lost. To estimate the
scale factor, further exploration is required, and some methods have been proposed
in literature [33, 34]. These methods require some form of ground truth information, such as the actual coordinates of 3D points or known dimensions in a scene.
The ground truth information is then used to impose constraints on the parameter
correction step during bundle adjustment. This approach ensures that the metric
scale is preserved, and the calibrated stereo camera can be used for accurate 3D
reconstruction.\n
We use the Zhang’s algorithm camera calibration parameters as an initial estimate of the intrinsic matrix
 for stereo calibration. However, there are other ways to
estimate intrinsic parameters, as described in literature such as [35, 36, 37]. 
The paper by Richard [38] proposes a method for estimating intrinsic and extrinsic parameters of a camera 
that only undergoes rotation through a mechanical mechanism,
without needing to consider the epipolar geometry of the scene. This approach can
be particularly useful in situations where the camera is fixed to a rotating platform,
such as in robotics or surveillance systems. By using this method, it is possible to
obtain accurate camera calibration results without the need for a complex 
calibration setup or extensive human intervention.\n
The accuracy of monitoring systems heavily relies on camera calibration. A
widely used calibration method is the checkerboard pattern, originally proposed by
Zhang. However, this approach involves human intervention, which can introduce
errors and be time-consuming. We can explore the self-calibration or autocalibra-
99tion technique that enables the camera system to estimate calibration parameters,
such as pose and intrinsic values, without any user input. In the context of traffic monitoring, 
Romil et al.[39] have proposed a deep learning-based system that
extracts key-point features from car images to estimate camera calibration parameters from a few hundred samples. 
This technique is helpful in estimating vehicle
speeds and issuing challans automatically in case of speed violations. Similarly, in
smart cities, thousands of outdoor cameras are deployed without calibration information, 
such as mounting height and orientation, which can limit their potential
use in surveillance and road planning. Viktor and Milan [40] have proposed 
a traffic surveillance camera calibration method based on detecting pairs of vanishing
points associated with vehicles in traffic surveillance footage.
''')
