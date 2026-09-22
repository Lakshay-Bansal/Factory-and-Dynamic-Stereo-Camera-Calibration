import streamlit as st
import os
import sys

PWD = os.getcwd()
sys.path.append(rf'{PWD}\Images')
st.title("Algorithms")

st.write("# Factory Camera Calibration")

st.write("### Stereo-Capture Multi-view Camera Calibration")

st.write('''
This algorithm is performed on the checkerboard pattern. 
The sequential steps involved in stereo capture multi-view camera calibration are depicted in Figure 4.1.
''')

st.image('Images/SCMVCC.png', 
         caption='''Figure 4.1: Steps of stereo capture multi-view camera calibration. ''',
                width=500)



st.write("# Dynamic Camera Calibration")
st.write("### Single Capture Dynamic Calibration Algorithm (SCDCA)")

st.write('''
This algorithm is performed on the dynamic scene images. 
Using photographs of the calibration pattern of known dimensions with different
descriptions like the number of rows, the number of columns, and the size of a
square in a checkerboard pattern a camera calibration can be carried out. 
This is a dynamic camera calibration performed with images of dynamic outdoor landscapes using a
single capture algorithm to calibrate the camera. The steps involved in the algorithm are shown in Fig. 4.2. 
''')

st.image('Images/singleCaptureAlgoSteps.jpg', 
         caption='''Figure 4.2: Flowchart of single capture dynamic calibration algorithm. ''',
                width=500)

st.write('''
A camera system is used to acquire multi-view images, and then features are
detected in the images. The detected features are then mapped to the corresponding features 
in different view images. After doing the essential matrix computation with the help of the 
five-point algorithm approach on the matched feature, 3D
world points are obtained from the matched feature. The best 3D points are then
chosen by bundle adjustment, which reduces the re-projection error. Additionally,
guided feature matching will be carried out to choose the best feature. The final
step in the camera calibration process is validation in which the 3D measurement of
the object in an image can be performed, to confirm that the camera parameters that
were determined using a single capture dynamic camera calibration are accurate.
''')


st.write("### Stereo-Capture Multi-view Dynamic Camera Calibration")

st.write('''
Fig. 4.3, presents the flowchart of multi-view dynamic stereo camera calibration
which takes single-capture dynamic calibration algorithm presented in Chapter 5
as a processing block in some step of the following algorithm. The reference image
is denoted as Iref , which is the most recently captured image frame set for which
dynamic calibration parameters need to be estimated. The pool of N candidate
image frame sets is denoted as Ii∈{1,...,N}.
''')

st.image('Images/flowchart_MultiCapt_Algo.jpg', 
         caption='''Figure 4.3: Flowchart of multi-view dynamic stereo camera calibration. ''',
                width=600)
