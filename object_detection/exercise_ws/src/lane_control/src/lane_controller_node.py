#!/usr/bin/env python3
import numpy as np
import rospy
import os
import yaml


from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import Point as PointMsg
from lane_controller.controller import PurePursuitLaneController
#from duckietown_utils import load_homography
from image_processing.ground_projection_geometry import Point, GroundProjectionGeometry
from image_processing.rectification import Rectify
from image_geometry import PinholeCameraModel

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
        self.params['~K_theta'] = rospy.get_param('~K_theta', None)
        self.params['~K_d'] = rospy.get_param('~K_d', None)
        #self.H = load_homography()
        #self.log(str(self.H))
        self.ground_projector = None
        self.rectifier = None
        self.homography = self.load_extrinsics()
        self.camera_info_received = False
        self.log(str(self.homography)) 
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
                                                 
        self.sub_camera_info = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/camera_node/camera_info", CameraInfo, self.cb_camera_info, queue_size=1)
        self.log("Lane controller initialized!")

    def cb_camera_info(self, msg):
        """
        Initializes a :py:class:`image_processing.GroundProjectionGeometry` object and a
        :py:class:`image_processing.Rectify` object for image rectification

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.

        """
        if not self.camera_info_received:
            self.rectifier = Rectify(msg)
            self.ground_projector = GroundProjectionGeometry(im_width=msg.width,
                                                         im_height=msg.height,
                                                         homography=np.array(self.homography).reshape((3, 3)))
        self.camera_info_received=True
        
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
        # self.log('sizes segment list')
        # self.log(str(len(w_seg)))
        # self.log(str(len(y_seg)))
        


    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages
        Computes v and omega using PPController

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header
        (v, omega) = self.pp_controller.computeControlAction(self.w_seg_list, self.y_seg_list, self.pose_msg.phi, self.pose_msg.d)
        car_control_msg.omega = omega
        car_control_msg.v = v
        # self.log('v et omega')
        # self.log(str(car_control_msg.v))
        # self.log(str(car_control_msg.omega))
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
    
    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.

        Returns:
            :obj:`numpy array`: the loaded homography matrix

        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file,'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']

if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
