#!/usr/bin/env python

from __future__ import division
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from controller import PostionController

#Marker
from utilities import MarkerServer, pointMarker
from visualization_msgs.msg import Marker
from path_generater.srv import UpdateMarker


def clicked_callback(point):
    pose = PoseStamped()
    pose.header = point.header
    pose.pose.position.x = point.point.x
    pose.pose.position.y = point.point.y
    pose.pose.orientation.w = 1.0
    rospy.loginfo("x: %.2f y: %.2f", point.point.x, point.point.y)
    controller.set_target(pose)
    updateTarget(name='target', marker=makePointMarker(pose, Marker.ADD))

rospy.init_node("task_track_point")
rate = rospy.Rate(20)
controller = PostionController((1.1, 0, 0), (1, 0.03, 0),max_angular_integral=100)

# Marker
markerServer = MarkerServer(topic_ns='target', service_ns='updateMarker')
makePointMarker = pointMarker(scale=[0.07,0.07,0.07], color=[0.2,0.7,0.2,1.0])
updateTarget = rospy.ServiceProxy(markerServer.service_name, UpdateMarker)

rospy.Subscriber("odom", Odometry, controller.odom_callback)
rospy.Subscriber("/clicked_point", PointStamped, clicked_callback)

pose = PoseStamped()
pose.header.frame_id='world'
pose.pose.position.x = 0.9
pose.pose.orientation.w = 1.0
controller.set_target(pose)

rate.sleep()
updateTarget.wait_for_service()
res = updateTarget(name='target', marker=makePointMarker(pose, Marker.ADD))
print(res)
 
while not rospy.is_shutdown():
    try:
        controller.update_cmd()
        rate.sleep()
    except (rospy.ROSInterruptException):
        updateTarget.close() 

    