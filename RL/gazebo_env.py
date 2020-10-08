'''
to run gazebo_env with python3:
source ~/catkin_ws_garyT/ros_py3/bin/activate
source ~/catkin_ws_garyT/devel_isolated/setup.bash 
to run gazebo simulator:
source ~/catkin_ws_garyT/devel/setup.bash 
OR (THIS MAY WORK)
source ~/catkin_ws_garyT/devel_isolated/setup.bash 
'''
import rospy
import tf
import math
from tf import transformations
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
import numpy as np
import time

class GazeboEnv():
    def __init__(self):
        self.robot_name = "mybot_0"
        rospy.init_node('env_node')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.robot_state = None # robot state is the model state obtained from GetModelState [header pose twist success status_message]
        self.state = None # state includes [d, alpha, v_{t-1}, w_{t-1}] which is based on IROS 2017
        self.goal = [0,0, np.pi]
        self.actions = dict(linear_vel=dict(shape=(), type='float', min_value=0.0, max_value=1.0),
                                 angular_vel=dict(shape=(), type='float', min_value=-1.0, max_value=1.0))
        self.vel_cmd = [0,0]
        self.v_max = 1
        self.w_max = 3.14
        self.time_step = 0.05
        self.reward_lamba = [0.5, 0.5] # hyperparameter for HIT 2019 (github) reward function
        self.cr = 20 # hyerparameter for IROS 2017 reward function
        self.k1 = 0.5
        self.k2 = 0.1
        self.d_previous = 0
        self.vel_previous = None
        self.count = 0
    
    def excute(self, action):
        done = False
        vel_cmd = Twist()
        vel_cmd.linear.x = self.v_max*action['linear_vel']
        vel_cmd.angular.z = self.w_max*action['angular_vel']
   
        self.vel_pub.publish(vel_cmd)
        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]

        time.sleep(self.time_step)
        rospy.wait_for_service('/gazebo/get_model_state')

        try:
            self.robot_state = self.get_state(self.robot_name, "world")
            # print(self.robot_state.pose.position)
        except rospy.ROSInterruptException as e:
            print("get robot pose fail!")
        
        d, alpha = self.cal_relative_pose()
        
        '''
        the one implemented based on high speed drifting, not pretty sure whether useful or not
        '''
        # linear_reward = math.exp(-self.k1*d)
        # angular_reward_func = lambda x: math.exp(-0.1*x) if abs(x) < 90 else( -math.exp(-0.1*(180-x)) if x >=90 else -math.exp(-0.1*(180+x)))
        # reward = linear_reward + angular_reward_func(alpha)
        '''    
        the one implemented based on Lei Tai 2017 IROS
        ''' 
        reward = self.cr*(self.d_previous - d)
        self.d_previous = d
       
        # reward = -self.reward_lamba[0]*abs(alpha) - self.reward_lamba[1]*d
        # reward for finishing the test
        if d < 0.1:
            done = True
            reward = 500
            self.count += 1
        # reward for moving into the virtual boundary
        if d > 10:
            done = True
            reward = -50
        self.state = np.array([d, alpha]+self.vel_cmd)
        return self.state, reward, done, {}


    
    def cal_relative_pose(self):
        orientation = self.robot_state.pose.orientation
        d_x = self.goal[0] - self.robot_state.pose.position.x
        d_y = self.goal[1] - self.robot_state.pose.position.y
        _, _, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        d, alpha = self.goal2robot(d_x, d_y, theta)
        if alpha > np.pi:
            alpha = alpha -2*np.pi
        if alpha < -np.pi:
            alpha = alpha + 2*np.pi
        return d, alpha

    def goal2robot(self, d_x, d_y, theta):
        d = math.sqrt(d_x * d_x + d_y * d_y)
        alpha = math.atan2(d_y, d_x) - theta
        return d, alpha

    def reset(self):
        # reset vel
        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)
        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]
        # reset robot pose
        start_position = np.random.uniform(0,2,2)
        start_angle = np.random.uniform(-np.pi, np.pi, 1)
        self.set_start(start_position[0], start_position[1], start_angle[0])
        # reset robot goal
        self.goal = [0,0, np.pi]
        d, alpha = self.cal_relative_pose()
        self.state = np.array([d, alpha]+self.vel_cmd)
        self.d_previous = d
        self.vel_previous = [vel_cmd.linear.x, vel_cmd.angular.z]
        return self.state


    def set_start(self, x, y, theta):
        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0
        quat = Quaternion(* transformations.quaternion_from_euler(0,0,theta))
        state_msg.pose.orientation = quat
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            resp = self.set_state(state_msg)
            self.robot_state = state_msg
        except rospy.ROSInterruptException as e:
            print("set robot start position fail!")

    def reset_test_goal(self):
        goal_list = [[0,0], [0,2], [2,2], [2,0]]
        self.goal = goal_list[self.count%4]+[np.pi]
        
    def set_goal(self):
        target_position = np.random.uniform(0,self.length,2)
        return [target_position[0],target_position[1], np.pi] 

    def kinetic_constraint():
        return

if __name__ == "__main__":
    env = GazeboEnv()
    env.reset()
    action = dict(linear_vel=0, angular_vel=0)
    # state,reward,done ,_ = env.excute(action)
    # print(env.state)
    '''
    Control with PID
    '''
    while 1:
        state,reward,done ,_ = env.excute(action)
        d = state[0]
        alpha = state[1]
        action["linear_vel"] = 1*d
        action["angular_vel"] = 2*alpha
        if done:
            action["linear_vel"] = 0
            action["angular_vel"] = 0
            env.reset()
        print(reward)