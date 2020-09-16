#!/usr/bin/env python

import rospy
import copy

from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import Marker, InteractiveMarkerControl, InteractiveMarker, InteractiveMarkerFeedback
from geometry_msgs.msg import Point, PoseStamped

def makeBox( msg ):
    marker = Marker()

    marker.type = Marker.CUBE
    marker.scale.x = msg.scale * 0.45
    marker.scale.y = msg.scale * 0.45
    marker.scale.z = msg.scale * 0.45
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 0.5
    marker.color.a = 1.0

    return marker

def makeTrafficCone( msg ):
    marker = Marker()

    marker.type = Marker.MESH_RESOURCE
    marker.scale.x = 0.008
    marker.scale.y = 0.008
    marker.scale.z = 0.006
    marker.color.r = 0.92
    marker.color.g = 0.768
    marker.color.b = 0.247
    marker.color.a = 1.0
    marker.mesh_resource = "package://path_generater/meshes/cone.obj"
    # marker.mesh_use_embedded_materials = True

    return marker


def normalizeQuaternion( quaternion_msg ):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm**(-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s

def makeObstacle(position):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "world"
    int_marker.pose.position = position
    int_marker.scale = 0.25

    int_marker.name = "obstacle"
    int_marker.description = "obstacle"

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0

    normalizeQuaternion(control.orientation)
    control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
    int_marker.controls.append(copy.deepcopy(control))

    # make a box which also moves in the plane
    control.markers.append( makeTrafficCone(int_marker) )
    control.always_visible = True
    int_marker.controls.append(control)

    # we want to use our special callback function
    server.insert(int_marker, processFeedback)

def processFeedback(feedback):
    if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        pose = feedback.pose
        rospy.loginfo( feedback.marker_name + str(pose.position.x) + "," 
                        + str(pose.position.y) + "," 
                        + str(feedback.pose.position.z) )
        
        posestamped = PoseStamped()
        posestamped.header = feedback.header
        posestamped.pose = pose
        point_pub.publish(posestamped)

if __name__=="__main__":
    rospy.init_node("obstacle_node")
    server = InteractiveMarkerServer("obstacle") 
    point_pub = rospy.Publisher('obstacle/pose', PoseStamped, queue_size=100)

    position = Point(0.9, 0.2, 0.0)
    makeObstacle(position)    

    server.applyChanges()

    rospy.spin()

