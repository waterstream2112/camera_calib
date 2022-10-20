
### Intrinsic calibration
rosrun camera_calibration cameracalibrator.py --no-service-check --size 9x6 --square 0.04 image:=/multijackal_01/d435i_2/color/image_raw camera:=/multijackal_01

rosrun camera_calibration cameracalibrator.py --no-service-check --size 9x6 --square 0.04 image:=/argus/ar0234_front_right/image_raw camera:=/argus

### Launch the dog set up file
--> launch either one of these two depending on which dog

roslaunch autonomous_robots_2 handheld_mapping_setup.launch
roslaunch autonomous_robots_2 handheld_mapping_setup_2.launch

### Extrinsic calibration
--> change camera and lidar input topic in launch file

roslaunch camera_calib cam_lidar_calib.launch

### Run projection
roslaunch camera_calib cam_lidar_proj_d435i.launch
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map multijackal_01 100
rosrun rviz rviz -f multijackal_01 -d `rospack find camera_calib`/draconis.rviz





# 
D = [0.20368984736396123, -0.5017791816281467, 0.00437797647196533, -0.0017745008471770705, 0.0]
K = [638.8067357735649, 0.0, 311.9925991638236, 0.0, 639.7254644805623, 230.59116584620043, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [650.6316528320312, 0.0, 310.46248133662084, 0.0, 0.0, 652.237548828125, 231.41912143059017, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
640

height
480

[narrow_stereo]

camera matrix
638.806736 0.000000 311.992599
0.000000 639.725464 230.591166
0.000000 0.000000 1.000000

distortion
0.203690 -0.501779 0.004378 -0.001775 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
650.631653 0.000000 310.462481 0.000000
0.000000 652.237549 231.419121 0.000000
0.000000 0.000000 1.000000 0.000000
--------------------------------------------------






# Procyon's D435_2 camera calibration data

D = [0.14070226683396006, -0.3693458181112717, 0.006314613071611573, 0.005215521798002149, 0.0]
K = [626.0765535002184, 0.0, 325.2746302000688, 0.0, 626.5313543466928, 234.8165871832428, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [632.6786499023438, 0.0, 327.51561137864337, 0.0, 0.0, 635.2723388671875, 236.23341164240992, 0.0, 0.0, 0.0, 1.0, 0.0]
None
(oST version 5.0 parameters)


[image]

width
640

height
480

[narrow_stereo]

camera matrix
626.076554 0.000000 325.274630
0.000000 626.531354 234.816587
0.000000 0.000000 1.000000

distortion
0.140702 -0.369346 0.006315 0.005216 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
632.678650 0.000000 327.515611 0.000000
0.000000 635.272339 236.233412 0.000000
0.000000 0.000000 1.000000 0.000000


### Dog2 intrinsic calibration

ono pinhole calibration...
D = [-0.19016897287498472, 0.02999339386545281, 0.0008888529134721215, 0.0005324680098406593, 0.0]
K = [411.8702384444886, 0.0, 463.4245039794622, 0.0, 416.30666581048575, 271.8247762668035, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [292.34674072265625, 0.0, 463.632088766346, 0.0, 0.0, 378.55633544921875, 272.3980910358114, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
960

height
540

[narrow_stereo]

camera matrix
411.870238 0.000000 463.424504
0.000000 416.306666 271.824776
0.000000 0.000000 1.000000

distortion
-0.190169 0.029993 0.000889 0.000532 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
292.346741 0.000000 463.632089 0.000000
0.000000 378.556335 272.398091 0.000000
0.000000 0.000000 1.000000 0.000000

