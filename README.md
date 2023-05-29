
### Intrinsic camera calibration
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


### workflow during calibration

- opencv window show image feed from camera
- filtered pointcloud is published
- plane pointcloud is published
- 



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


### Dog2 argus_front_right intrinsic calibration

ono pinhole calibration...
D = [-0.20682036837982382, 0.037210806349556536, 0.0010772408967416461, 0.00023617350820891265, 0.0]
K = [396.95262289269954, 0.0, 493.3202096639713, 0.0, 397.91876890579937, 272.2465312036496, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [291.7879638671875, 0.0, 490.3217159794294, 0.0, 0.0, 354.51312255859375, 273.28491488687723, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
960

height
540

[narrow_stereo]

camera matrix
396.952623 0.000000 493.320210
0.000000 397.918769 272.246531
0.000000 0.000000 1.000000

distortion
-0.206820 0.037211 0.001077 0.000236 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
291.787964 0.000000 490.321716 0.000000
0.000000 354.513123 273.284915 0.000000
0.000000 0.000000 1.000000 0.000000


### Dog2 d435 intrinsic calibration
ono pinhole calibration...
D = [0.2208048093839422, -0.5519158257372787, -0.002710753977377344, 0.00013723943088670202, 0.0]
K = [634.1102575192741, 0.0, 315.20791449060596, 0.0, 633.7205406020618, 227.86751508693214, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [646.4725341796875, 0.0, 314.6524384604927, 0.0, 0.0, 646.1294555664062, 226.57113628251864, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
640

height
480

[narrow_stereo]

camera matrix
634.110258 0.000000 315.207914
0.000000 633.720541 227.867515
0.000000 0.000000 1.000000

distortion
0.220805 -0.551916 -0.002711 0.000137 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
646.472534 0.000000 314.652438 0.000000
0.000000 646.129456 226.571136 0.000000
0.000000 0.000000 1.000000 0.000000
