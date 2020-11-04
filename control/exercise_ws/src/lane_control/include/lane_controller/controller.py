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
        
    def compute_control_action(self, listedessegments, Llim, lanewidth, K):
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
    	follow_point = self.find_following_point(listedesegments, Llim, lanewidth)
    	L = np.sqrt(follow_point[0]**2+follow_point[1]**2)
    	sin_alpha = follow_point[1]/L
    	v = 0.5
    	omega = sin_alpha/K
    	
    	return(v,omega)
    		
    def find_following_point(self, segments, Llim, lanewidth):
    	""" Idee : a partir de la liste des segments tries grace au node lane_filter_node (fonction get_inlier_segments), on trouve un segment de ligne jaune qui nous va (a une distance sup a Llim) et on trouve le point au milieu de la voie (a une distance delta_d du milieu du segment suivant la normale au segment)
    		parametres : - self
    		             - listedessements : segment msg published by lane_filter_node
    		             - Llim : distance de visibilite prescrite
    		             - lanewidth : largeur de la voie (pour calculer delta_d)
    		sorie :      - follow_point : array avec les coordonnees du points a suivre
    	"""
    	delta_d = lanewidth/2
    	follow_point = np.array([0.0,0.0])
    	length_max = 0
    	for segment in segments:
    		p1 = np.array([segment.points[0].x, segment.points[0].y])
	        p2 = np.array([segment.points[1].x, segment.points[1].y])
        	t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
		n_hat = np.array([-t_hat[1], t_hat[0]])
		if segment.color == segment.YELLOW: # left lane is yellow
			length = sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
			if length>length_max: # we take the longest segment
				x_c = (p1[0]+p2[0])/2
				y_c = (p1[1]+p2[1])/2
				dist = sqrt(x_c**2+y_c**2)
				if dist>=Llim: #check to see if it is ahead enough
					if (p2[0]<p1[0]): #right side of the yellow lane
						follow_point[0] = x_c+n_hat[0]*lane_width/2
						follow_point[1] = x_c+n_hat[1]*lane_width/2
						length_max = length
	return(follow_point)
	

    def get_normal_vector(self,segment):
    	p1 = np.array([segment.points[0].x, segment.points[0].y])
    	p2 = np.array([segment.points[1].x, segment.points[1].y])
    	t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    	if segment.color == segment.YELLOW:
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
    	elif segment.color == segment.WHITE:
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
    	return(n_hat)
    	
    def get_center_segment(self, segment):
    	x_c = (segment.points[0].x+segment.points[1].x)/2
    	y_c = (segment.points[0].y+segment.points[1].y)/2
    	center = np.array([x_c,y_c])
    	return(center)
    	
    def get_distance(self,point):
    	d = sqrt(point[0]**2+point[0]**2)
    	return(d)
    	
	


