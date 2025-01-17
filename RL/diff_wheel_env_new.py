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
        self.robot_name = "robot1/mybot"
        rospy.init_node('env_node')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.vel_pub = rospy.Publisher("/robot1/cmd_vel", Twist, queue_size=10)
        self.robot_state = None # robot state is the model state obtained from GetModelState [header pose twist success status_message]
        self.state = None # state includes [d, alpha, v_{t-1}, w_{t-1}] which is based on IROS 2017
        self.goal = [0,0, np.pi]
        self.actions = dict(linear_vel=dict(shape=(), type='float'),
                                 angular_vel=dict(shape=(), type='float'))
        self.vel_cmd = [0,0]
        self.v_max = 10
        self.v_car_max = 1.5
        self.w_max = 3.14
        self.time_step = 0.05
        self.reward_lamba = [0.5, 0.5] # hyperparameter for HIT 2019 (github) reward function
        self.cr = 200 # hyerparameter for IROS 2017 reward function
        self.k1 = 20
        self.k2 = 20
        self.k3 = [20,20]
        self.d_previous = 0
        self.vel_previous = [0, 0]
        self.count = 0
        self.pitch = 0
        
        # unicycle model parameter
        self.r = 0.059
        self.l = 0.12
        self.J = np.array([[self.r/2, self.r/2],[self.r/self.l, -self.r/self.l]])

    def excute(self, action):
        done = False
        vel_cmd = Twist()
        v = np.array([self.v_max*action['linear_vel'], self.v_max*action['angular_vel']])
        self.vel_cmd = np.dot(self.J, v)
        if self.vel_cmd[0] > 1.1:
            self.vel_cmd[0] = 1.1
        if self.vel_cmd[1] > 3.14:
            self.vel_cmd[1] = 3.14
        vel_cmd.linear.x = self.vel_cmd[0]
        vel_cmd.angular.z = self.vel_cmd[1]
    
        self.vel_pub.publish(vel_cmd)

        
        # print("v type: {}".format(type(vel_cmd.linear.x)))
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
        reward = self.cr*(self.d_previous - d) - self.k3[0]*abs(self.vel_cmd[0]-self.vel_previous[0]) -self.k3[1]*abs(self.vel_cmd[1]-self.vel_previous[1])
        reward = np.float(reward)
        self.d_previous = d
        self.vel_previous = [self.vel_cmd[0], self.vel_cmd[1]]
        # reward = -self.reward_lamba[0]*abs(alpha) - self.reward_lamba[1]*d
        # reward for finishing the test
        # punish the agent if the termial speed is not zero
        if d < 0.1:
            done = True
            reward = 200 - self.k1*self.vel_cmd[0] - self.k2*self.vel_cmd[1]
            reward = np.float(reward)
            # print(type(reward))
            self.count += 1
        # reward for moving into the virtual boundary
        if d > 2.75:
            done = True
            reward = -50
        # if abs(self.pitch) > 0.1:
        #     done = True
        #     reward = -50
            # self.v_car_max *= 0.99
            # self.w_max *= 0.99
            # if self.v_car_max < 1:
            #     self.v_car_max = 1
            # if self.w_max < 2.8:
            #     self.w_max = 2.8
        self.state = np.array([d, alpha]+list(self.vel_cmd))
        # print(self.vel_cmd)
        # print(self.state)
        return self.state, reward, done, [self.robot_state.pose.position.x, self.robot_state.pose.position.y]


    
    def cal_relative_pose(self):
        orientation = self.robot_state.pose.orientation
        d_x = self.goal[0] - self.robot_state.pose.position.x
        d_y = self.goal[1] - self.robot_state.pose.position.y
        roll, self.pitch, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        # print("roll is{}".format(roll))
        # print("pitch is{}".format(pitch))
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
        self.goal = [1,1, np.pi]
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
    xx = env.reset()
    print(xx)
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