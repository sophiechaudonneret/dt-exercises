#!/usr/bin/env python3
import numpy as np
import rospy
import os

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
        ~seglist_filtered (:obj:`SegmentList`): The detected line segments from the line detector filtered to keep only the 'good' ones
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params['~look_ahead_distance'] = rospy.get_param('~look_ahead_distance', None)
        self.params['~K'] = rospy.get_param('~K', None)
        self.K_theta = rospy.get_param('~K_theta', None)
        self.K_d = rospy.get_param('~K_d', None)

        self.pp_controller = PurePursuitLaneController(self.params)
        self.log('Controller Publisher...')
        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)
        self.log('Controller Subscribers...')
        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)

        self.sub_segment_list = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/lane_filter_node/seglist_filtered",
                                                 SegmentList,
                                                 self.cbSegmentList,
                                                 queue_size=1)
        self.log("Lane controller initialized!")


    def cbSegmentList(self, seglist):
        """Callback receiving filtered segments list messages

        Args:
            seglist(:obj:`SegmentList`): The detected line segments from the line detector filtered to keep only the 'good' ones
        """
        w_seg = []
        y_seg = []
        for segment in seglist.segments:
            if segment.color == segment.WHITE: # white lane
                w_seg.append(segment)
            elif segment.color == segment.YELLOW: # yellow lane
                y_seg.append(segment)
        self.w_seg_list = w_seg
        self.y_seg_list = y_seg
        self.log('sizes segment list')
        self.log(str(len(w_seg)))
        self.log(str(len(y_seg)))
        


    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages
        Computes v and omega using PPController

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header
        (v, omega) = self.pp_controller.computeControlAction(self.w_seg_list, self.y_seg_list)
        # v = 0.1
        # omega = 0
        if omega is not None:
            car_control_msg.omega = omega
            car_control_msg.v = v
        else : # proportional controller
            car_control_msg.v = v / 4 # v_barre
            car_control_msg.omega = self.K_theta* self.pose_msg.phi + self.K_d * self.pose_msg.d
        self.log('v et omega')
        self.log(str(car_control_msg.v))
        self.log(str(car_control_msg.omega))
        self.publishCmd(car_control_msg)


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.pp_controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
