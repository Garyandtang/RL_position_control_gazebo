#!/usr/bin/env python

from __future__ import division
import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

#path Publisher
from geometry_msgs.msg import Quaternion
from tf import transformations
from path_publisher import PathPublisher

def interpolate_two_point(a, b, num):
    a = np.array(a)
    b = np.array(b)
    diff = b - a
    norm = np.linalg.norm(diff)
    t = np.linspace(0, 1, num=num)
    return [list(a+c*diff) for c in t]

def update_waypoints(posestamped):
    global waypoints
    global trajectoryPub

    radius = 0.2
    obstacle_origin = np.array([posestamped.pose.position.x, posestamped.pose.position.y])

    updated_waypoints = []
    for p in waypoints:
        diff = np.array(p) - obstacle_origin
        norm = np.linalg.norm(diff)

        if norm < radius:
            p = diff/norm*radius + obstacle_origin

        updated_waypoints.append(list(p))

    trajectoryPub.updateWaypoints(updated_waypoints)


rospy.init_node("task_trajectory_generation")

rate = rospy.Rate(20)

waypoints = interpolate_two_point([0.9, 0.8], [0.9, -0.5], 50)
trajectoryPub = PathPublisher(waypoints, frame_id='world', topic_name='global_path',
                                publishTF=False)
obstacle_pose_sub = rospy.Subscriber('obstacle/pose', PoseStamped, update_waypoints)

rospy.loginfo("Task: %s start.", rospy.get_name()) 
while not rospy.is_shutdown():
    try:
        trajectoryPub.publish()
        rate.sleep()
    except (rospy.ROSInterruptException):
       pass
