#!/usr/bin/env python

from __future__ import division
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion
from utilities import MarkerServer, pointMarker
import numpy as np

from path_generater.srv import UpdateMarker
from visualization_msgs.msg import Marker
from path_generater.srv import UpdateMarker, UpdateMarkerResponse

class PostionController(object):
    def __init__(self, linear_coeff, angular_coeff, max_angular_integral = 100 ,max_linear_integral = 2, prefix=''):
        self._globalPose = PoseStamped()
        self._target = PoseStamped()
        self._velPub = rospy.Publisher(prefix + "cmd_vel", Twist, queue_size=10)

        self._linear_P = linear_coeff[0]
        self._linear_I = linear_coeff[1]
        self._linear_D = linear_coeff[2]

        self._angular_P = angular_coeff[0]
        self._angular_I = angular_coeff[1]
        self._angular_D = angular_coeff[2]

        self._max_angular_integral = max_angular_integral
        self._max_linear_integral = max_linear_integral

        self._a_err_integral = 0
        self._l_err_integral = 0

    @property
    def pose(self):
        return self._globalPose
        
    @property
    def position(self):
        return self._globalPose.pose.position
        
    @property
    def quaternion(self):
         return self._globalPose.pose.orientation

    @property
    def euler(self):
        q = [self.quaternion.x, self.quaternion.y, self.quaternion.z, self.quaternion.w]
        return euler_from_quaternion(q)
    
    def odom_callback(self, odom):
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self._globalPose = pose
    
    def set_target(self, targetPose):
        self._target = targetPose

    def update_cmd(self, stop=False):
        if stop:
            twist = Twist()
            self._velPub.publish(twist)
            return

        else:
            theta_target = np.arctan2(self._target.pose.position.y - self.position.y, self._target.pose.position.x - self.position.x)
            angular_err = theta_target - self.euler[2]
            while np.abs(angular_err) > np.pi:
                if angular_err > 0:
                    angular_err = angular_err - 2 * np.pi
                else:
                    angular_err = angular_err + 2 * np.pi
            linear_err = np.sqrt((self._target.pose.position.y - self.position.y)**2
                               + (self._target.pose.position.x - self.position.x)**2) * np.cos(angular_err)
            
            self._a_err_integral = self._a_err_integral + angular_err
            self._l_err_integral = self._l_err_integral + linear_err
            if np.abs(self._a_err_integral) > self._max_angular_integral:
                self._a_err_integral = self._max_angular_integral
            
            if np.abs(self._l_err_integral) > self._max_linear_integral:
                self._l_err_integral = self._max_linear_integral
            
            twist = Twist()

            twist.linear.x = self._linear_P * linear_err + self._linear_I * self._l_err_integral
            twist.angular.z = self._angular_P * angular_err + self._angular_I * self._a_err_integral
            self._velPub.publish(twist)




if __name__ == "__main__":
    controller = PostionController((1, 0, 0), (1, 0, 0))
    rospy.init_node("controller")
    rospy.Subscriber("odom", Odometry, controller.odom_callback)
    markerServer = MarkerServer(topic_ns='target', service_ns='updateMarker')
    updateTarget = rospy.ServiceProxy(markerServer.service_name, UpdateMarker)
    rate = rospy.Rate(50)

    makePointMarker = pointMarker(scale=[0.1,0.1,0.1], color=[0.2,0.7,0.2,1.0])
    pose = PoseStamped()
    pose.header.frame_id='world'
    pose.pose.orientation.w = 1.0
    pose.pose.position.x = 0.9
    pose.pose.position.y = 0.9

    while not rospy.is_shutdown():
        try:
            pose.pose.position.x = pose.pose.position.x + 0.001
            try:
                updateTarget(name='target', marker=makePointMarker(pose, Marker.ADD))
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s"%e)

            controller.set_target(pose)
            rospy.loginfo("Target: %.2f %.2f", pose.pose.position.x, pose.pose.position.y)
            controller.update_cmd()
            rate.sleep()
            
        except rospy.ROSInterruptException:
            rospy.logwarn("STOP")
            controller.update_cmd(stop=True)
            updateTarget.close()
    