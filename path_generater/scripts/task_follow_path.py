#!/usr/bin/env python

from __future__ import division
import rospy
import numpy as np
from local_planner import LocalPlanner
from controller import PostionController
from nav_msgs.msg import Path, Odometry

#Marker
from utilities import MarkerServer, pointMarker
from visualization_msgs.msg import Marker
from path_generater.srv import UpdateMarker

#path Publisher
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Quaternion
from tf import transformations
from eight_ellipse import eight_ellipse
from path_publisher import PathPublisher

rospy.init_node("task_follow_path")

# controller
controller = PostionController((1.1, 0, 0), (1, 0.03, 0),max_angular_integral=100)

#planner
planner = LocalPlanner(loop = True, look_ahead_index=3)

# Marker
markerServer = MarkerServer(topic_ns='target', service_ns='updateMarker')
makePointMarker = pointMarker(scale=[0.07,0.07,0.07], color=[0.2,0.7,0.2,1.0])
updateTarget = rospy.ServiceProxy(markerServer.service_name, UpdateMarker)

#tf
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

#Path publisher
world2camera = tf_buffer.lookup_transform('world', 'camera_link', rospy.Time(0), timeout=rospy.Duration(0.02))
world2camera.transform.translation.z = 0.0
world2camera.transform.rotation = Quaternion(* transformations.quaternion_from_euler(0,0,-np.pi/2))
waypoints = eight_ellipse(0.3, 0.65, 0.5, arc_step = 0.03, circle_step=0.2, ellipse_step = 0.1)
eightEllipsePub = PathPublisher(waypoints, frame_id='path_predefined', topic_name='global_path',
                                     publishTF=True, reference_frame='world', transform=world2camera.transform)

###
rospy.Subscriber("odom", Odometry, controller.odom_callback)
rospy.Subscriber("global_path", Path, planner.update_global_plan)

rate = rospy.Rate(20)

rospy.loginfo("Task: %s start.", rospy.get_name()) 
while not rospy.is_shutdown():
    try:
        eightEllipsePub.publish()

        if controller.pose.header.frame_id == '':
            continue
        target = planner.make_plan(controller.pose)
        target = tf_buffer.transform(target, 'world')
        updateTarget(name='target', marker=makePointMarker(target, Marker.ADD))
        
        controller.set_target(target)
        controller.update_cmd()
        rate.sleep()
    except tf2_ros.TransformException as e:
        rospy.logerr(e)
    except (rospy.ROSInterruptException):
        updateTarget.close()

