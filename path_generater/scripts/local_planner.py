#!/usr/bin/env python

from __future__ import division
import rospy
import tf2_ros as tf
import tf2_geometry_msgs
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class LocalPlanner(object):
    def __init__(self, loop = False, look_ahead_index = 2, look_ahead_bound = 10):
        self._loop = loop
        self._look_ahead_index = look_ahead_index
        self._look_ahead_bound = look_ahead_bound
        self.poses = []
        self.tfBuffer  = tf.Buffer()
        self.tfListener = tf.TransformListener(self.tfBuffer)
        self.curIndex = 0

    def update_global_plan(self, path):
        self.poses = path.poses
        # print(len(self.poses))

    def make_plan(self, current_pose):
        pose_array = self.poses
        if len(pose_array) == 0:
            return current_pose

        try:
            current_pose = self.tfBuffer.transform(current_pose, pose_array[0].header.frame_id,timeout=rospy.Duration(0.02))
        except (tf.ConnectivityException, tf.LookupException, tf.ExtrapolationException) as e:
            rospy.logerr("Err: From %s to %s | %s", current_pose.header.frame_id, pose_array[0].header.frame_id, e)
            return current_pose

        min_distance = np.inf
        min_index = 0
        
        bound_index = self.curIndex + self._look_ahead_bound
        if bound_index >= len(pose_array):
            bound_index = -1
        
        for index, p in enumerate(pose_array[self.curIndex : bound_index]):
            dis = (p.pose.position.x - current_pose.pose.position.x) ** 2 + (p.pose.position.y - current_pose.pose.position.y) ** 2
            if dis < min_distance:
                min_index = index
                min_distance = dis

        self.curIndex = min_index + self.curIndex
        target_index = self.curIndex + self._look_ahead_index
       
        if target_index >= len(pose_array):
            if not self._loop:
                return pose_array[-1]
            else:
                self.curIndex = target_index % len(pose_array)
                return pose_array[target_index % len(pose_array)]
        else:
            return pose_array[target_index]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from eight import eight

    rospy.init_node("local_planner", anonymous=True)
    planner = LocalPlanner(loop=True, look_ahead_index=0)
    pose_list = eight(0.3)
    x,y = zip(*pose_list)
    plt.scatter(x,y)
    plt.axis('equal')

    path = Path()
    for index, p in enumerate(pose_list):
        pose = PoseStamped()
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = 0
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    
    planner.update_global_plan(path)

    pose = PoseStamped()
    pose.pose.orientation.w = 1.0
    pose.pose.position.x = 0.3
    pose.pose.position.y = 0.2
    plt.plot(pose.pose.position.x, pose.pose.position.y, 'go')
    t_pose = planner.make_plan(pose)
    plt.plot(t_pose.pose.position.x, t_pose.pose.position.y, 'ro')
    plt.show()


