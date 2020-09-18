# How to using ROS with Python3

To obtain the camera's intrinsic parameter and distortion coefficients

#### for general stuff

这个命令安装python3 ros基本库。没有tf和cv bridge
```bash
sudo apt-get install python3-catkin-pkg-modules
sudo apt-get install python3-rospkg-modules
```



### for tf

 http://community.bwbot.org/topic/499/%E5%9C%A8ros%E4%B8%AD%E4%BD%BF%E7%94%A8python3

**remark**:

git clone geometry and geometry2 by default是melodic-devel和noetic-devel, ubuntu 16.04需要换成indigo-devel

我的virtualenv虚拟环境就在catkin_sim_ws下面
source ~/catkin_sim_ws/venv/bin/activate
```bash
catkin_make_isolated --cmake-args                      -DCMAKE_BUILD_TYPE=Release
```
try this command
```bash
catkin_make_isolated --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
 ```
 

#### pip3

pip3也是有奇奇怪怪的问题等再遇见了把这个补全

