#!/usr/bin/env python

from __future__ import division
import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from eight_ellipse import eight_ellipse
import tf2_ros
from tf import transformations

class PathPublisher(object):
    def __init__(self, waypoints = [], frame_id='', topic_name='', publishTF = False, reference_frame='', transform = None):
        assert not (publishTF and (frame_id == '' or transform == None or reference_frame=='')), "Please specify frame_id & reference_frame & transform"
       
        self._path = Path()
        self._frame_id = frame_id
        self._topic_name = topic_name
        self._pub = rospy.Publisher(self._topic_name, Path, queue_size=1)
        self._publishTF = publishTF
        self._reference_frame = reference_frame
        self.transform = transform

        self.updateWaypoints(waypoints)

        if publishTF:
            self._tf_broad = tf2_ros.StaticTransformBroadcaster()

            transformStamped = TransformStamped()
            transformStamped.header.stamp = rospy.Time.now()
            transformStamped.header.frame_id = self._reference_frame
            transformStamped.child_frame_id = self._frame_id = frame_id
            transformStamped.transform = transform

            self._tf_broad.sendTransform(transformStamped)

    def updateWaypoints(self, waypoints):
        self._waypoints = waypoints
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self._frame_id

        for index, p in enumerate(self._waypoints):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = 0
            if index == 0:
                pose.pose.orientation = Quaternion(*transformations.quaternion_from_euler(0,0,0))
            else:
                theta = np.arctan2(p[1] - self._waypoints[index - 1][1], p[0] - self._waypoints[index - 1][0])
                pose.pose.orientation = Quaternion(*transformations.quaternion_from_euler(0,0,theta))
            path.poses.append(pose)
        
        if len(path.poses) >= 2:
            path.poses[0].pose.orientation = path.poses[1].pose.orientation
            
        self._path = path

    def publish(self):
        self._path.header.stamp = rospy.Time.now()
        self._pub.publish(self._path)
        
if __name__ == "__main__":
    rospy.init_node('path_publisher')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    world2camera = tf_buffer.lookup_transform('world', 'camera_link', rospy.Time(0), timeout=rospy.Duration(0.02))
    world2camera.transform.translation.z = 0.0
    world2camera.transform.rotation = Quaternion(* transformations.quaternion_from_euler(0,0,-np.pi/2))

    waypoints = eight_ellipse(0.3, 0.65, 0.5, arc_step = 0.03, circle_step=0.2, ellipse_step = 0.1)
    eightEllipsePub = PathPublisher(waypoints, frame_id='path_predefined', topic_name='global_path',
                                     publishTF=True, reference_frame='world', transform=world2camera.transform)
        
    rate = rospy.Rate(50)
    
    while not rospy.is_shutdown():
        try:
            eightEllipsePub.publish()
            rospy.loginfo("%d poses have been published", len(waypoints))
            rate.sleep()
        except (rospy.ROSInterruptException) as e:
            rospy.logerr("Path publisher: %s", e)

            


