
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt, pi



class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = np.array([[self.mean_d_0], [self.mean_phi_0]])
        self.cov_0 = np.array([[self.sigma_d_0, 0], [0, self.sigma_phi_0]])

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0.0
        self.initialized = False
        self.Q = np.array([[0.5, 0], [0, 0.5]]) # Q is the noise covariance matrix --> to be tuned
        self.matrix = 0

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics
        if not self.initialized:
            return
        d_belief = self.belief['mean'][0][0]
        phi_belief = self.belief['mean'][1][0]
        cov_belief = self.belief['covariance']
        alpha = self.encoder_resolution / (2*pi)
        F = self.Fmatrix(alpha, left_encoder_delta, right_encoder_delta)
        L = self.Lmatrix(alpha)
        d_predicted = d_belief + np.sin(phi_belief) * self.wheel_radius * 0.5 * (left_encoder_delta + right_encoder_delta) / alpha
        phi_predicted = phi_belief + self.wheel_radius * (left_encoder_delta - right_encoder_delta) / (self.baseline * alpha)
        cov_predicted = F.dot(cov_belief.dot(F.transpose())) + L.dot(self.Q.dot(L.transpose()))
        self.belief['mean'] = np.array([[d_predicted], [phi_predicted]])
        self.belief['covariance'] = cov_predicted


    def Fmatrix(self, alpha, left_encoder_data, right_encoder_data):
        F = np.eye((2))
        F[0, 1] = np.cos(self.belief['mean'][1][0]) * self.wheel_radius * (left_encoder_data + right_encoder_data) * 0.5 / alpha
        return F

    def Lmatrix(self, alpha):
        L = np.zeros((2, 2))
        L[0, 0] = np.cos(self.belief['mean'][1][0]) * self.wheel_radius * 0.5 / alpha
        L[0, 1] = np.cos(self.belief['mean'][1][0]) * self.wheel_radius * 0.5 / alpha
        L[1, 0] = self.wheel_radius / (self.baseline * alpha)
        L[1, 1] = self.wheel_radius / (self.baseline * alpha)
        return L

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)
        # self.matrix = measurement_likelihood
        # parameter measurement as gaussian --> find mean and covariance matrix
        if measurement_likelihood is not None:
            # mean is the maximum value in the histogram
            maxids = np.unravel_index(measurement_likelihood.argmax(), measurement_likelihood.shape)  # find max index
            d_mean = self.d_min + (maxids[0] + 0.5) * self.delta_d  # compute corresponding d
            phi_mean = self.phi_min + (maxids[1] + 0.5) * self.delta_phi  # compute corresponding phi
            mean_vec = np.array([[d_mean], [phi_mean]])  # mean vector --> our measured estimate
            # sigma = self.getSigmaEstimate(measurement_likelihood, mean_vec)  # our estimated noise matrix
            i_min, j_min, i_max, j_max = self.defineBondarySigma(measurement_likelihood, maxids)
            sigma = self.getSigmaMatrix(measurement_likelihood, mean_vec, i_min, j_min, i_max, j_max)
            if np.linalg.matrix_rank(sigma) != 2:
                sigma = np.array([[1, 0.1], [0.1, 1]])
            self.matrix = measurement_likelihood

            # TODO: Apply the update equations for the Kalman Filter to self.belief
            X_predicted = self.belief['mean']
            P_predicted = self.belief['covariance']
            H = np.eye((2))
            M = np.eye((2))
            V = H.dot(P_predicted.dot(H.transpose())) + M.dot(sigma.dot(M.transpose()))
            K = P_predicted.dot(np.dot(H.transpose(), np.linalg.inv(V)))
            X_updated = X_predicted + K.dot((mean_vec-X_predicted))
            P_updated = P_predicted - K.dot(H.dot(P_predicted))
            self.belief['mean'] = X_updated
            # self.belief['mean'] = mean_vec
            self.belief['covariance'] = P_updated
            # return V.shape(), K.shape(), P_predicted.shape()
            return d_mean, phi_mean, sigma
        else:
            return None, None, None

    def getSizes(self):
        return np.shape(self.matrix)

    # def defineBondarySigma(self, distribution_matrix, maxids):
    #     # mean is the maximum value in the histogram
    #     window_size = [5, 5]
    #     matrix_size = np.shape(distribution_matrix)
    #     new_matrix = np.zeros_like(distribution_matrix)
    #     new_matrix[max(maxids[0]-window_size[0],0):min(maxids[0]+window_size[0],matrix_size[0]),max(maxids[1]-window_size[1],0):min(maxids[1]+window_size[1],matrix_size[1])] \
    #         = distribution_matrix[max(maxids[0]-window_size[0],0):min(maxids[0]+window_size[0],matrix_size[0]),max(maxids[1]-window_size[1],0):min(maxids[1]+window_size[1],matrix_size[1])]
    #     return new_matrix

    def defineBondarySigma(self, distribution_matrix, maxids):
        # mean is the maximum value in the histogram
        window_size = [5, 5]
        matrix_size = np.shape(distribution_matrix)
        i_min = max(maxids[0]-window_size[0],0)
        j_min = max(maxids[1]-window_size[1],0)
        i_max = min(maxids[0]+window_size[0],matrix_size[0])
        j_max = min(maxids[1]+window_size[1],matrix_size[1])
        return i_min, j_min, i_max, j_max

    # def getSigmaEstimate(self, distribution_matrix, mean_value):
    #     sigma = self.getSigmaMatrix(distribution_matrix, mean_value)
    #     if np.linalg.matrix_rank(sigma) == 2:
    #         num_outliers = 0
    #         outliers = 100
    #         while outliers > 0:
    #             distribution_matrix, outliers = self.RejectOutliersMahalanobis(distribution_matrix, mean_value, sigma)
    #             num_outliers += outliers
    #             sigma_temp = self.getSigmaMatrix(distribution_matrix, mean_value)
    #             if np.linalg.matrix_rank(sigma_temp)==2:
    #                 sigma = sigma_temp
    #     else:
    #         # only case occupied --> not enough data --> high covariance matrix to trust the wheel odometry
    #         sigma = np.array([[100, 0], [0, 100]])
    #     return sigma

    # def getSigmaMatrix(self, distribution_matrix, mean_value):
    #     sigma = np.zeros((2, 2))
    #     for i in range(distribution_matrix.shape[0]):
    #         d_i = self.d_min + (i + 0.5) * self.delta_d
    #         for j in range(distribution_matrix.shape[1]):
    #             phi_i = self.phi_min + (j + 0.5) * self.delta_phi
    #             X = np.array([[d_i], [phi_i]])-mean_value
    #             sigma += distribution_matrix[i, j] * X.dot(np.transpose(X))
    #     sigma = sigma/np.sum(distribution_matrix)
    #     return sigma

    def getSigmaMatrix(self, distribution_matrix, mean_value, i_min, j_min, i_max, j_max):
        sigma = np.zeros((2, 2))
        for i in range(i_min, i_max):
            d_i = self.d_min + (i + 0.5) * self.delta_d
            for j in range(j_min, j_max):
                phi_i = self.phi_min + (j + 0.5) * self.delta_phi
                X = np.array([[d_i], [phi_i]])-mean_value
                sigma += distribution_matrix[i, j] * X.dot(np.transpose(X))
        sigma = sigma/np.sum(distribution_matrix[i_min:i_max,j_min:j_max])
        return sigma
    # def RejectOutliersMahalanobis(self, distribution_matrix, mean_value, sigma):
    #     outliers = 0
    #     for i in range(distribution_matrix.shape[0]):
    #         d_i = self.d_min + (i + 0.5) * self.delta_d
    #         for j in range(distribution_matrix.shape[1]):
    #             phi_i = self.phi_min + (j + 0.5) * self.delta_phi
    #             if distribution_matrix[i, j] != 0:
    #                 X = np.array([[d_i], [phi_i]]) - mean_value
    #                 Mahal = np.dot(np.transpose(X), np.dot(np.linalg.inv(sigma), X))
    #                 if Mahal > 12:
    #                     distribution_matrix[i, j] = 0
    #                     outliers += 1
    #     return distribution_matrix, outliers

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)

        return measurement_likelihood





    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray