#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg
import os
import yaml

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, AntiInstagramThresholds
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import Point as PointMsg
from image_processing.anti_instagram import AntiInstagram
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge
from image_processing.ground_projection_geometry import Point, GroundProjectionGeometry
from image_processing.rectification import Rectify
from image_geometry import PinholeCameraModel


class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )



        # Construct publishers
        self.pub_obj_dets = rospy.Publisher(
            "~duckie_detected",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.sub_thresholds = rospy.Subscriber(
            "~thresholds",
            AntiInstagramThresholds,
            self.thresholds_cb,
            queue_size=1
        )
        
        self.sub_camera_info = rospy.Subscriber(
            f"/{os.environ['VEHICLE_NAME']}/camera_node/camera_info", 
            CameraInfo, 
            self.cb_camera_info, 
            queue_size=1
        )
        
        self.sub_lane_reading = rospy.Subscriber(
            f"/{os.environ['VEHICLE_NAME']}/lane_filter_node/lane_pose", 
            LanePose, 
            self.cbLanePoses, 
            queue_size=1
        )

        self.initialized = False
        self.ai_thresholds_received = False
        self.anti_instagram_thresholds=dict()
        self.ai = AntiInstagram()
        self.bridge = CvBridge()

        self.ground_projector = None
        self.rectifier = None
        self.homography = self.load_extrinsics()
        self.camera_info_received = False
        self.log(str(self.homography)) 
        self.lane_width = rospy.get_param('~lanewidth', None)
        self.safe_distance = rospy.get_param('~safe_distance', None)

        model_file = rospy.get_param('~model_file','.')
        rospack = rospkg.RosPack()
        model_file_absolute = rospack.get_path('object_detection') + model_file
        self.model_wrapper = Wrapper(model_file_absolute)
        self.initialized = True
        self.image_count = 0
        self.obstacle_left_lane = False
        self.obstacle_right_lane = False
        self.log("Initialized!")
    
    def thresholds_cb(self, thresh_msg):
        self.anti_instagram_thresholds["lower"] = thresh_msg.low
        self.anti_instagram_thresholds["higher"] = thresh_msg.high
        self.ai_thresholds_received = True

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
            self.im_width=msg.width
            self.im_height=msg.height

        self.camera_info_received=True
    
    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages
        Computes v and omega using PPController
        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        # TODO to get better hz, you might want to only call your wrapper's predict function only once ever 4-5 images?
        # This way, you're not calling the model again for two practically identical images. Experiment to find a good number of skipped
        # images.

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        
        # Perform color correction
        if self.ai_thresholds_received:
            image = self.ai.apply_color_balance(
                self.anti_instagram_thresholds["lower"],
                self.anti_instagram_thresholds["higher"],
                image
            )
        
        image = cv2.resize(image, (224,224))
        if self.image_count != 0:
            self.image_count = np.mod(self.image_count+1, 3)
        else :
            bboxes, classes, scores = self.model_wrapper.predict(image)
            im_boxed = self.plotWithBoundingBoxes(image,bboxes[0],classes[0],scores[0])
            cv2.imshow('detected objects', im_boxed)
            cv2.waitKey(1)
            self.det2bool(bboxes[0], classes[0]) # [0] because our batch size given to the wrapper is 1
        
        msg = BoolStamped()
        msg.header = image_msg.header
        if self.obstacle_right_lane:
            msg.data = True
            if self.obstacle_left_lane :
                pass
            else :
                ## OVERTAKING
                # msg.data = overtake 
                pass
        
        
        self.pub_obj_dets.publish(msg)
    
    def det2bool(self, bboxes, classes):
        # TODO remove these debugging prints
        # print(bboxes)
        # print(classes)
        
        # This is a dummy solution, remove this next line
        # return len(bboxes) > 1
    
        
        # TODO filter the predictions: the environment here is a bit different versus the data collection environment, and your model might output a bit
        # of noise. For example, you might see a bunch of predictions with x1=223.4 and x2=224, which makes
        # no sense. You should remove these. 
        
        # TODO also filter detections which are outside of the road, or too far away from the bot. Only return True when there's a pedestrian (aka a duckie)
        # in front of the bot, which you know the bot will have to avoid. A good heuristic would be "if centroid of bounding box is in the center of the image, 
        # assume duckie is in the road" and "if bouding box's area is more than X pixels, assume duckie is close to us"
        
        
        self.obstacle_right_lane = False
        self.obstacle_left_lane = False
        obj_det_list = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            label = classes[i]
            if label == 1 :
                if (x2-x1 >= 2) and (y2-y1 >= 2) : 
                    low_center = Point((x1 + x2)/2, y2)
                    rect_pixel = self.rectifier.rectify_point(low_center)
                    ground_point = self.ground_projector.pixel2ground(rect_pixel)
                    duckie_lane_pose = np.cos(self.pose_msg.phi)*(ground_point.y+self.pose_msg.d)+np.sin(self.pose_msg.phi)*ground_point.x
                    dist = np.sqrt(ground_point.x**2 + ground_point.y**2)
                    if np.abs(duckie_lane_pose)<=self.lane_width/2: #in our lane
                        if dist <= self.safe_distance:
                            self.obstacle_right_lane = True
                    elif np.abs(duckie_lane_pose)>self.lane_width/2: #in left lane
                        if dist <= self.safe_distance*1.5:
                            self.obstacle_left_lane = True
            # TODO if label isn't a duckie, skip
            # TODO if detection is a pedestrian in front of us:
            #   return True

    def plotWithBoundingBoxes(self,seg_im,boxes,labels,scores):
        for i in range(len(labels)):
            cv2.rectangle(seg_im, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), (255,255,255),1)
            cv2.rectangle(seg_im,(boxes[i][0],boxes[i][1]),(boxes[i][0]+20,boxes[i][1]-6),(255,255,255),cv.FILLED)
            cv2.putText(seg_im,f"{labels[i]} : {scores[i]}",(boxes[i][0],boxes[i][1]),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        return seg_im

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
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
