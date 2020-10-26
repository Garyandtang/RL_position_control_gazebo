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

from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import itertools
from scipy.special import comb
class GazeboEnv():
    def __init__(self, robot_name = "robot1/mybot", robot_no = 2):
        self.robot_name = robot_name
        self.robot_index = int(robot_name.split('/')[0][-1])
        self.robot_no = robot_no
        rospy.init_node('env_node')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        vel_topic_name = "{}/cmd_vel".format(self.robot_name.split('/')[0])
        self.vel_pub = rospy.Publisher(vel_topic_name, Twist, queue_size=10)
        self.robot_state = None # robot state is the model state obtained from GetModelState [header pose twist success status_message]
        self.state = None # state includes [d, alpha, v_{t-1}, w_{t-1}] which is based on IROS 2017
        self.goal = [0,0, np.pi]
        self.positions_list = np.zeros((robot_no,2))
        self.actions = dict(linear_vel=dict(shape=(), type='float'),
                                 angular_vel=dict(shape=(), type='float'))
        self.vel_cmd = [0,0]
        self.v_max = 10 # max wheel linear velocity
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
        # time.sleep(self.time_step)
        rospy.wait_for_service('/gazebo/get_model_state')

        for i in range(self.robot_no):
            robot_name = "robot{}/mybot".format(i+1)
            # print(robot_name)
            if robot_name == self.robot_name:
                try:
                    robot_state = self.get_state(robot_name, "world")
                    self.positions_list[i] = [robot_state.pose.position.x, robot_state.pose.position.y]
                    self.robot_state = robot_state
                except rospy.ROSInterruptException as e:
                    print("get robot pose fail!")
            else:
                try:
                    robot_state = self.get_state(robot_name, "world")
                    self.positions_list[i] = [robot_state.pose.position.x, robot_state.pose.position.y]
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
        # # deminish w_max and v_max, to find best w_max and v_max
        # if abs(self.pitch) > 0.1:
        #     done = True
        #     reward = -50
        #     self.v_car_max *= 0.99
        #     self.w_max *= 0.99
        #     if self.v_car_max < 1:
        #         self.v_car_max = 1
        #     if self.w_max < 2.8:
        #         self.w_max = 2.8
        self.state = np.array([d, alpha]+list(self.vel_cmd))
        orientation = self.robot_state.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        return self.state, reward, done, self.positions_list, theta


    
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
        # self.set_start(start_position[0], start_position[1], start_angle[0])
        self.set_start(1, 0, 0)
        # reset robot goal
        self.goal = [3,1, np.pi]
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

    # predefined trajectory a 2x2 square
    def reset_test_goal(self):
        goal_list = [[0,0], [0,2], [2,2], [2,0]]
        self.goal = goal_list[self.count%4]+[np.pi]

    # randomly reset goal   
    def set_goal(self):
        target_position = np.random.uniform(0,self.length,2)
        return [target_position[0],target_position[1], np.pi] 

    def kinetic_constraint():
        return

def si_to_uni_dyi(dx, theta):
    linear_velocity_gain = 1
    angular_velocity_limit=np.pi
    a = np.cos(theta)
    b = np.sin(theta)
    dxu = np.zeros((2,1))
    dxu[0] = linear_velocity_gain*(a*dx[0]+b*dx[1])
    dxu[1] = angular_velocity_limit*np.arctan2(-b*dx[0] + a*dx[1], dxu[0])/(np.pi/2)
    return dxu

def si_to_uni_dyi_2(dx, theta):
    linear_velocity_gain = 1
    angular_velocity_limit=np.pi
    a = np.cos(theta)
    b = np.sin(theta)
    l = 0.01
    dxu = np.zeros((2,1))
    dxu[0] = a*dxu[0] + b*dxu[1] 
    dxu[1] = -1/l*b*dxu[0] + 1/l*a*dx[1]
    return dxu

def uni_to_si_dyi(dxu, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    l = 0.01
    dx = np.zeros((2,1))
    dx[0] = a*dxu[0] - l*b*dxu[1] 
    dx[1] = b*dxu[0] + l*a*dx[1]
    return dx

if __name__ == "__main__":
    env = GazeboEnv()
    xx = env.reset()
    print(xx)
    action = dict(linear_vel=0, angular_vel=0)

    r = 0.059
    l = 0.12
    J = np.array([[r/2, r/2],[r/l, -r/l]])
    J_inv = np.linalg.inv(J)
    v_max = 0
    w_max = 0
    '''
    Control with PID
    '''
    while 1:
        state,reward,done ,positions_list, theta = env.excute(action)
        # print(positions_list)
        d = state[0]
        alpha = state[1]
        v = np.array([0.1*d, 0.16*alpha])
      
        # dx = np.zeros((2,1))
        # dx[0] = v[0]*np.cos(theta)
        # dx[1] = v[0]*np.sin(theta)
        dx = uni_to_si_dyi(v, theta)
        print(v)
        print(dx)
        '''
        barrier function implementation
        here only consider one control input
        how to map unicycle control input to si control input is a problem!!
        '''
        N = positions_list.shape[0]
        num_constraints = int(comb(N,2))
        A = np.zeros((num_constraints, N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(N)))
        count = 0
        safety_radius=0.1
        barrier_gain=100
        for i in range(N-1):
            for j in range(i+1, N):
                error = positions_list[i,:] - positions_list[j,:]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1))] = -2*error
                # A[count, (2*j, (2*j+1))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1
                
        f = -2*np.reshape(dx, N, order='F')
        qp_result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        qp_result = np.reshape(qp_result, (2, -1), order='F')
        qp_uni_v = si_to_uni_dyi(dx, theta)
        print(qp_uni_v)

        v_wheel = np.dot(J_inv, qp_uni_v)
        action["linear_vel"] = v_wheel[0]
        action["angular_vel"] = v_wheel[1]
        print(action)
        if done:
            action["linear_vel"] = 0
            action["angular_vel"] = 0
            env.reset()
        # print(reward)
        # if v[0] > v_max:
        #     v_max = v[0]
        # if v[1] > w_max:
        #     w_max = v[1]
        # print([v_max, w_max])