import numpy as np


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters
        self.look_ahead_distance = self.parameters['~look_ahead_distance']
        self.K = self.parameters['~K']
        self.K_d = self.parameters['~K_d']
        self.K_theta = self.parameters['~K_theta']
        self.enoughData = True
        self.v_purepursuit = 0.6
        self.v_prop = 0.2
        


    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def getCenterSegment(self,segment):
        x_c = (segment.points[0].x+ segment.points[1].x)/2
        y_c = (segment.points[0].y+ segment.points[1].y)/2
        center = [x_c, y_c]
        return center
    
    def getMiddlePoint(self, w_point, y_point):
        # actually not the middle as the bot tends to go to the white lane (don't know why)
        x_middle = 0.4 * w_point[0] + 0.6 * y_point[0]
        y_middle = 0.4 * w_point[1] + 0.6 * y_point[1]
        middle = [x_middle, y_middle]
        return middle

    def getDistance(self, point):
        dist = np.sqrt(point[0] ** 2 + point[1] ** 2)
        return dist

    def getPotentialFollowPoints(self, w_seg, y_seg):
        w_pot_fp = []
        for segment in w_seg :
            w_center = self.getCenterSegment(segment)
            w_pot_fp.append(w_center)
        y_pot_fp = []
        for segment in y_seg :
            y_center = self.getCenterSegment(segment)
            y_pot_fp.append(y_center)
        return (w_pot_fp, y_pot_fp)
            


    def getFollowPoint(self, w_pot_fp, y_pot_fp):
        if len(w_pot_fp) >= 2 and len(y_pot_fp) >= 2 :
            self.EnoughData = True
            wfp = [0,0]
            dist = 10
            for point in w_pot_fp :
                dist_temp = np.abs(self.getDistance(point) - self.look_ahead_distance)
                if dist_temp <= dist :
                    dist = dist_temp
                    wfp = point
            yfp = [0,0]
            dist = 10
            for point in y_pot_fp :
                dist_temp = np.abs(self.getDistance(point) - self.look_ahead_distance)
                if dist_temp <= dist :
                    dist = dist_temp
                    yfp = point
            fp = self.getMiddlePoint(wfp,yfp)
            dist = self.getDistance(fp)
            return fp, dist
        else:
            self.EnoughData = False
            return None, None
        

    def computeControlAction(self, w_seg, y_seg, phi, d):
        w_pot_fp, y_pot_fp = self.getPotentialFollowPoints(w_seg, y_seg)
        (fp, dist) = self.getFollowPoint(w_pot_fp, y_pot_fp)
        if self.EnoughData :
            sin_alpha = fp[1] / dist
            v = self.v_purepursuit
            omega = (sin_alpha / self.K)
        else :
            v = self.v_prop
            omega = self.K_theta* phi + self.K_d * d
        return v, omega