import rospy 
import rospkg 
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

def main():
    rospy.init_node('set_pose')

    state_msg = ModelState() 
    state_msg.model_name = 'mybot_0'
    state_msg.pose.position.x = 0.5
    state_msg.pose.position.y = 0.5
    state_msg.pose.position.z = 0
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 1

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    except (rospy.ROSInterruptException) as e:
        print("Service call failed: %s", e)
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # set_state( state_msg )
if __name__ == '__main__':
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()