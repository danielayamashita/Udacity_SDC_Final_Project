# Capstone project: Programming a real self-driving car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[overview]: /imgs/overview_graph.png "Overview" 
[tl_node]: /imgs/tl_node.png "Traffic Light" 
[waypoint_node]: /imgs/waypoint_node.png "Waypoint updater" 
[dbw_node]: /imgs/dbw_node.png "DBW" 

In this final project of the udacity self-driving car nanodegree, the full control software is written for a simplified driving situation. The code is tested on a simulator and on the Udacity self-driving car Carla at a test site in Palo Alto, California.
The car can maneuver on a highway or parking lot with traffic lights. The simplified situation does not include other vehicles, traffic signs or unexpected obstacles. How to handle such situations can be found in other projects.

## The team

| Name | GitHub profile |
|------|----------------|
| Daniela | [danielayamashita](https://github.com/danielayamashita)|
| Vinod | [imvinod](https://github.com/imvinod) |
| Wen | [WenHsu1203](https://github.com/WenHsu1203) |
| Markus | [Macki767](https://github.com/Macki767) |
| Solvejg | [heliotropium72](https://github.com/heliotropium72) |

## System architecture

Carla is controlled using the robot operating system ([ROS](http://www.ros.org/)).

![alt text][overview]


### Waypoint node

![alt text][waypoint_node]


The waypoint updater extracts the next x waypoints from the `/base_waypoints` along the road which the car should follow. The creation is based on the current position, the target velocity and traffic lights.

More details...

### Drive-by-wire node

![alt text][dbw_node]

The throttle, brake and steering of Carla have electric control. The values which are send to Carla are created with a PID controller. They are published at 50Hz. This frequency is important because at lower frequency Carla might detect a fault and the system might shut down due to security settings.

In both the simulator and Carla the driver has to be able to take over manual control at any moment. In the code this is considered by the `dbw_enabled` flaged. The dbw control is only active when the flag is True.


### Traffic light detection and classification node

![alt text][tl_node]

#### Data sets
Some data sets were taken from the excellent report from level-5-engineers for this project. [Check it out here](https://github.com/level5-engineers/system-integration/wiki/Traffic-Lights-Detection-and-Classification)

Data from the simulator, provided by our peer level-5-engineers
[Simulator data](https://www.dropbox.com/s/87xark39qyer8df/TLdataset02.zip)
The data set contains 4499 images classified in 4 folders (red: 1733, yellow: 253, green: 645, unknown: 1868), every image is of size 224x224

Real road data:
[Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)

For myself:
( Terminal download with wget <path_to_folder.zip> and unzip it with unzip  TLdataset02.zip -d TLdataset02 )
(For loading the traffic light data I had to install sklearn, the training is independent of carla. So installing this library should not have a consequence)

#### Model
I retrained the last 15 layers of a mobilenet CNN. MobileNet a small models with few weights (ca 17MB of weights). It is supposed to be faster but less accurate in comparison to other CNNs.

## Simulator and test site

two testing environments in the simulator
test site in Paolo Alto, California

## Carla

Carla is an autonomous Lincoln MKZ with the following hardware specifications:
- 31.4 GiB Memory
- Intel Core i7-6700K CPU @ 4 GHz x 8
- TITAN X Graphics
- 64-bit OS


## Challenges and future improvements

----

## Information from Udacity

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
