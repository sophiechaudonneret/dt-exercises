import numpy as np


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters
        
    def compute_control_action(self, listedessegments, Llim, lanewidth, K, threshold):
    	­"""Idee : 
    		parametres : - self
    		             - listedessements : segment msg published by lane_filter_node
    		             - Llim : distance de visibilite prescrite
    		             - lanewidth : largeur de la voie (pour calculer delta_d)
    		             - K : parametre du filtre
    		sorie :      - v : la vitesse longitudinale du robot (pour l'instant constante)
    		             - omega : vitesse angulaire du robot
    		trouver un point au milieu de la voie ­a l'aide d'un segment
    		puis faire ppoursuite
    		peut etre faut il faire une fonction en elle meme pour trouver le point?
    	­"""
    	if len(listedesegments)!= 0: 
	    	follow_point = self.find_following_point(listedesegments, Llim, lanewidth, threshold)
    		L = np.sqrt(follow_point[0]**2+follow_point[1]**2)
    		sin_alpha = follow_point[1]/L
    		v = 0.5
    		omega = sin_alpha/K
    	else:
    		v = 0
    		omega = 0
    	return(v,omega)
    		
    def find_following_point(self, segments, Llim, lanewidth, threshold):
    	""" Idee : a partir de la liste des segments tries grace au node lane_filter_node (fonction get_inlier_segments), on trouve un segment de ligne jaune qui nous va (a une distance sup a Llim) et on trouve le point au milieu de la voie (a une distance delta_d du milieu du segment suivant la normale au segment)
    		parametres : - self
    		             - listedessements : segment msg published by lane_filter_node
    		             - Llim : distance de visibilite prescrite
    		             - lanewidth : largeur de la voie (pour calculer delta_d)
    		sorie :      - follow_point : array avec les coordonnees du points a suivre
    	"""
    	follow_point = np.array([0.0,0.0])
    	potential_fp, dist = get_potential_fp(segment,lanewidth)
    	delta_dist = np.abs(dist - Llim)
    	delta_dist_sorted = np.sort(delta_dist)
    	idx = np.argsort(delta_dist)
    	follow_points = potential_fp[idx]
    	i = 0
    	while delta_dist_sorted[i]<=threshold: 
    		follow_point = follow_point + follow_points[i]
    		i=i+1
    	follow_point = follow_point/(i+1)
	return follow_point
	

    def normal_vector_segment(self,segment):
    	p1 = np.array([segment.points[0].x, segment.points[0].y])
    	p2 = np.array([segment.points[1].x, segment.points[1].y])
    	t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    	if segment.color == segment.YELLOW: #left lane is yellow
    		if t_hat[0]<t_hat[1]: #line is more along the x axis (unlikelym only in turns)
    			if t_hat[0]>0:
    				n_hat = np.array([-t_hat[1], t_hat[0]])
    			else:
    				n_hat = np.array([t_hat[1], -t_hat[0]])
    		else: #line is along the y axis (more likely, straight lines)
    			if t_hat[1]>0:
    				n_hat = np.array([t_hat[1], -t_hat[0]])
    			else:
    				n_hat = np.array([-t_hat[1], t_hat[0]])
    	elif segment.color == segment.WHITE: #right lane is white
    		if t_hat[0]<t_hat[1]: #line is more along the x axis (unlikelym only in turns)
    			if t_hat[0]>0:
    				n_hat = np.array([t_hat[1], -t_hat[0]])
    			else:
    				n_hat = np.array([-t_hat[1], t_hat[0]])
    		else: #line is along the y axis (more likely, straight lines)
    			if t_hat[1]>0:
    				n_hat = np.array([-t_hat[1], t_hat[0]])
    			else:
    				n_hat = np.array([t_hat[1], -t_hat[0]])
    	else:
    		n_hat = None
    	return n_hat
    	
    def center_segment(self, segment):
    	x_c = (segment.points[0].x+segment.points[1].x)/2
    	y_c = (segment.points[0].y+segment.points[1].y)/2
    	center = np.array([x_c,y_c])
    	return center
    	
    def distance_point(self,point):
    	d = sqrt(point[0]**2+point[0]**2)
    	return d
    	
    def get_potential_fp(self, segments, lanewidth):
	potential_fp = []
	dist = []
	for segment in segments:
		n_hat = normal_vertor_segment(segment)
		if n_hat not None:
			center = center_segment(segment)
			fp = center + n_hat*lanewidth/2
			d = distance_point(fp)
			potential_fp.append(fp)
			dist.append(d)
	return potential_fp, dist
	


