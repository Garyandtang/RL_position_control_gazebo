## Gazebo Simulation for HKUST Swarm Project

A Gazebo simulation for differential wheeled robot swarm project. This work is based on:

* [differential wheeled robot](https://github.com/yangliu28/swarm_robot_ros_sim/blob/master/swarm_robot_description/urdf/two_wheel_robot.urdf)
* [Gazebo Apriltag](https://github.com/koide3/gazebo_apriltag)
* [rrbot](https://github.com/ros-simulation/gazebo_ros_demos)
* [Apriltag_localization](https://github.com/Swarm-UST/Apriltag_localization)
* [Turtlebot3_Keyboard_Control](https://github.com/ROBOTIS-GIT/turtlebot3)
<p align="center"><img src="https://github.com/Swarm-UST/gazebo_ros_demos/blob/master/documents/images/demo.jpg" class="center"></p>

### What's in this repo?

- [x] A differential wheeled robot model with Apriltag attached
- [x] A Gazebo world with monocular camera tracking system
- [x] Visualization result of in image_view
- [x] Visualization result of Apriltag localization in RVIZ
- [x] Keyboard control
- [ ] (Further) Formation control



### Tasks to do
* [ ] **DEBUG APRILTAG TRANSFORMATION (VERY URGENT)**
* [ ] TF between world frame and camera link
* [ ] differential wheeled robot rviz model
- [x] single launch file for all needed tasks

### Requirements

* Ubuntu 16.04 with ROS Kinetic (Desktop-Full version)

  (Or you can install Gazebo, RVIZ and other requirements separately)

* [ ] (**haven't done**) (We will also test whether it works on Ubuntu 18.04 with ROS Melodic )

* [ ] (Optional (**haven't done**)) `ros-controller` to control the joint of camera holder 

  ```
  sudo apt-get install ros-kinetic-ros-control ros-kinetic-ros-controllers
  ```

  can follow [this](http://gazebosim.org/tutorials/?tut=ros_control) tutorial to learn how to control the joint in Gazebo with `ros-controller`. 

### Usage

#### Compile

1. clone this repo the src of your catkin workspace

2. (Optional if you want to export by hand) add export to your ~/.bashrc 

   ```
   gedit ~/.bashrc
   ```

   add this:

   ```
   export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/home/eeuser/catkin_ws/src/tag_robot_simulation/src/mybot_description/models
   ```

   for example in my case:

   ```
   export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:/home/eeuser/catkin_ws/src/tag_robot_simulation/src/mybot_description/models
   ```

   

3. build the repo in catkin workspace

   ```
   cd ~/catkin_ws/
   catkin_make
   ```

#### run with single launch file

1. open terminal and
	```
	roscore
	```

2. open another terminal and 

   ```
   source ~/catkin_ws/devel/setup.bash
   roslaunch rrbot_gazebo single_robot_simulation.launch 
   ```


#### Show Gazebo simulation

1. open terminal and
	```
	roscore
	```

2. open another terminal and 

   ```
   source ~/catkin_ws/devel/setup.bash
   roslaunch rrbot_gazebo rrbot_world.launch 
   ```

#### Show image view result

1. open terminal and

   ```
   rosrun image_view image_view image:=/rrbot/camera1/image_raw
   ```


#### Keyboard control

1. open terminal and

   ```
   export TURTLEBOT3_MODEL=burger
   roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
   ```




### Image

**Differential wheeled robot model**
<p align="center"><img src="https://github.com/Swarm-UST/gazebo_ros_demos/blob/master/documents/images/differentical_wheeled_robot.jpg" height="450"></p>

**Gazebo simulation platform**
<p align="center"><img src="https://github.com/Swarm-UST/gazebo_ros_demos/blob/master/documents/images/gazebo.png" height="450" class="center"></p>

**Image view result**
<p align="center"><img src="https://github.com/Swarm-UST/gazebo_ros_demos/blob/master/documents/images/image_view.png" height="450" class="center"></p>

### Demo video

[Keyboard control](https://www.youtube.com/watch?v=9PPc-BmptM8)

### Folder Structure
```
.
├── apriltags2          # apriltag package built with c
│   ├── CMakeLists.txt
│   ├── include
│   ├── package.xml
│   └── src
├── apriltags2_ros      # apriltag ros interface package 
│   ├── CMakeLists.txt
│   ├── config
│   ├── include
│   ├── launch
│   ├── msg
│   ├── nodelet_plugins.xml
│   ├── package.xml
│   ├── rviz
│   ├── scripts
│   ├── src
│   └── srv
├── documents           # supporting documents and images
│   └── images
├── mybot_description
│   ├── CMakeLists.txt
│   ├── launch
│   ├── models          # apriltag gazebo materials (texture)
│   ├── package.xml
│   └── urdf            # differential wheeled robot urdf model
├── readme.md
├── rrbot_description
│   ├── CMakeLists.txt
│   ├── launch
│   ├── meshes
│   ├── package.xml
│   └── urdf            # rrbot (camera holder) urdf model
├── rrbot_gazebo
│   ├── CMakeLists.txt
│   ├── launch          # launch files for simulation
│   ├── package.xml
│   └── worlds          # gazebo world
└── turtlebot3_teleop   # keyboard control package
    ├── CHANGELOG.rst
    ├── CMakeLists.txt
    ├── launch
    ├── nodes
    ├── package.xml
    ├── setup.py
    └── src
```
