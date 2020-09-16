import rospy
from tf import transformations
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
import numpy as np

class GazeboEnv():
    def __init__(self):
        rospy.init_node('env_node')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.goal = [1,1]
        self._actions = dict(linear_vel=dict(shape=(), type='float', min_value=0.0, max_value=1.0),
                                 angular_vel=dict(shape=(), type='float', min_value=-1.0, max_value=1.0))
    def excute(self, action):
        return

    def reset(self):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """
        # reset robot pose
        start_position = np.random.uniform(0,2,2)
        start_angle = np.random.uniform(-np.pi, np.pi, 1)
        start_pose = self.set_start(start_position[0], start_position[1], start_angle[0])
        # reset robot goal
        self.goal = [1,1]
        return


    def set_start(self, x, y, theta):
        state_msg = ModelState()
        state_msg.model_name = 'mybot_0'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0
        quat = Quaternion(* transformations.quaternion_from_euler(0,0,theta))
        state_msg.pose.orientation = quat
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            resp = self.set_state(state_msg)
            return state_msg.pose
        except rospy.ROSInterruptException as e:
            print("set robot start position fail!")


    def set_goal(self):
        return

    def kinetic_constraint():
        return

if __name__ == "__main__":
    env = GazeboEnv()
    env.reset()