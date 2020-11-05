#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.controller = PurePursuitLaneController(self.params)
        self.K = 0.2
        self.Llim = 0.05
        self.lanewidth = 0.23
        self.threshold = 0.02
        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
#        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
#                                                 LanePose,
#                                                 self.cbLanePoses,
#                                                 queue_size=1)
        
        self.sub_filtered_seg_list = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered",
        						 SegmentList,
        						 self.cbSegListFiltered, 
        						 queue_size=1)

        self.log("Initialized!")


    def cbSegListFiltered(self, segment_list_msg):
        """Callback receiving filtered segment list message
        Args:
            segment_list_msg (:obj:`SegmentList`): message containing a list of filtered valid segments.
        """
        self.segment_msg = segment_list_msg
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.segment_msg.header

        (v, omega) = self.controller.compute_control_action(self.segment_msg, self.Llim, self.lanewidth, self.K, self.threshold)
#        v = 1
#        omega = 0
        car_control_msg.v = v
        car_control_msg.omega = omega
        self.publishCmd(car_control_msg)

    
#    def cbLanePoses(self, input_pose_msg):
#        """Callback receiving pose messages
#
#        Args:
#            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
#        """
#        self.pose_msg = input_pose_msg
#
#        car_control_msg = Twist2DStamped()
#        car_control_msg.header = self.pose_msg.header
#
#        # TODO This needs to get changed
##        car_control_msg.v = 0.5
##        car_control_msg.omega = 0
#	inlier_segments = self.filter.get_inlier_segments(segment_list_msg.segments,
#                                                              d_max,
#                                                              phi_max)
#        inlier_segments_msg = SegmentList()
#        inlier_segments_msg.header = segment_list_msg.header
#        inlier_segments_msg.segments = inlier_segments
#        v, omega = self.controller.compute_control_action(self, parametresaajouter)
#        car.control_msg.v = v
#       car.control_msg.omega = omega
#
#        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
