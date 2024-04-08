from pose import *
import numpy as np
import networkx as nx

class BodyGraph:
    def __init__(self, 
                 pose_input,
                 anatomical_edges=True,
                 major_location_edges=True,
                 hand2hand_edges=True,
                 ):
        
        self.pose_input = pose_input
        self.body_graph = nx.MultiGraph()

        self.anatomical_edges = anatomical_edges
        self.major_location_edges = major_location_edges
        self.hand2hand_edges = hand2hand_edges


    def calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def generate_edge(self, node1, node2, part_1, part_2, add_z=False):
        """
        Generate edge between two nodes in the graph
        
        Args:
            node1: int
            node2: int
            part_1: str
            part_2: str
            add_z: bool
        
        Returns:
        
        """
        edges = {
            'x': self.pose_input[part_1][node1][0] - self.pose_input[part_2][node2][0],
            'y': self.pose_input[part_1][node1][1] - self.pose_input[part_2][node2][1],
        }

        if add_z: edges['z'] = self.pose_input[part_1][node1][2] - self.pose_input[part_1][node2][2]

        self.body_graph.add_edge(
            node1, node2, edges
        )

    def generate_hand_skeleton(self):
        """
        Generate the hand skeleton graph via the hand landmarks
        
        """

        if self.anatomical_edges:
            for relation in HAND_GRAPH_ANATOMICAL:
                self.generate_edge(relation[0], relation[1], 'RIGHT_HAND', 'RIGHT_HAND')
                self.generate_edge(relation[0], relation[1], 'LEFT_HAND', 'LEFT_HAND')

            for relation in BODY_GRAPH_ANATOMICAL:
                self.generate_edge(relation[0], relation[1], 'POSE', 'POSE')

        if self.major_location_edges:
            for relation in HAND_GRAPH_RELATIONAL:
                self.generate_edge(relation[0], relation[1], 'RIGHT_HAND', 'POSE')
                self.generate_edge(relation[0], relation[1], 'LEFT_HAND', 'POSE')
        
        if self.hand2hand_edges:
            for relation in HAND_2_HAND_RELATIONAL:
                self.generate_edge(relation[0], relation[1], 'RIGHT_HAND', 'RIGHT_HAND')
                self.generate_edge(relation[0], relation[1], 'LEFT_HAND', 'LEFT_HAND')

        return self.body_graph


HAND_GRAPH_ANATOMICAL = [

    (HAND_LANDMARKS.WRIST, HAND_LANDMARKS.THUMB_CMC),
    (HAND_LANDMARKS.THUMB_CMC, HAND_LANDMARKS.THUMB_MCP),
    (HAND_LANDMARKS.THUMB_MCP, HAND_LANDMARKS.THUMB_IP),
    (HAND_LANDMARKS.THUMB_IP, HAND_LANDMARKS.THUMB_TIP),

    (HAND_LANDMARKS.WRIST, HAND_LANDMARKS.INDEX_FINGER_MCP),
    (HAND_LANDMARKS.INDEX_FINGER_MCP, HAND_LANDMARKS.INDEX_FINGER_PIP),
    (HAND_LANDMARKS.INDEX_FINGER_PIP, HAND_LANDMARKS.INDEX_FINGER_DIP),
    (HAND_LANDMARKS.INDEX_FINGER_DIP, HAND_LANDMARKS.INDEX_FINGER_TIP),

    (HAND_LANDMARKS.WRIST, HAND_LANDMARKS.MIDDLE_FINGER_MCP),
    (HAND_LANDMARKS.MIDDLE_FINGER_MCP, HAND_LANDMARKS.MIDDLE_FINGER_PIP),
    (HAND_LANDMARKS.MIDDLE_FINGER_PIP, HAND_LANDMARKS.MIDDLE_FINGER_DIP),
    (HAND_LANDMARKS.MIDDLE_FINGER_DIP, HAND_LANDMARKS.MIDDLE_FINGER_TIP),

    (HAND_LANDMARKS.WRIST, HAND_LANDMARKS.RING_FINGER_MCP),
    (HAND_LANDMARKS.RING_FINGER_MCP, HAND_LANDMARKS.RING_FINGER_PIP),
    (HAND_LANDMARKS.RING_FINGER_PIP, HAND_LANDMARKS.RING_FINGER_DIP),
    (HAND_LANDMARKS.RING_FINGER_DIP, HAND_LANDMARKS.RING_FINGER_TIP),

    (HAND_LANDMARKS.WRIST, HAND_LANDMARKS.PINKY_MCP),
    (HAND_LANDMARKS.PINKY_MCP, HAND_LANDMARKS.PINKY_PIP),
    (HAND_LANDMARKS.PINKY_PIP, HAND_LANDMARKS.PINKY_DIP),
    (HAND_LANDMARKS.PINKY_DIP, HAND_LANDMARKS.PINKY_TIP)
]

BODY_GRAPH_ANATOMICAL = [

    # HEAD
    (BODY_LANDMARKS.LEFT_EYE, BODY_LANDMARKS.RIGHT_EYE),
    (BODY_LANDMARKS.LEFT_EYE, BODY_LANDMARKS.NOSE),
    (BODY_LANDMARKS.RIGHT_EYE, BODY_LANDMARKS.NOSE),
    (BODY_LANDMARKS.LEFT_EYE, BODY_LANDMARKS.LEFT_EAR),
    (BODY_LANDMARKS.RIGHT_EYE, BODY_LANDMARKS.RIGHT_EAR),
    (BODY_LANDMARKS.LEFT_EAR, BODY_LANDMARKS.NOSE),
    (BODY_LANDMARKS.RIGHT_EAR, BODY_LANDMARKS.NOSE),

    # CENTRAL
    (BODY_LANDMARKS.LEFT_SHOULDER, BODY_LANDMARKS.RIGHT_SHOULDER),
    (BODY_LANDMARKS.LEFT_SHOULDER, BODY_LANDMARKS.LEFT_ELBOW),
    (BODY_LANDMARKS.RIGHT_SHOULDER, BODY_LANDMARKS.RIGHT_ELBOW),
    (BODY_LANDMARKS.LEFT_ELBOW, BODY_LANDMARKS.LEFT_WRIST),
    (BODY_LANDMARKS.RIGHT_ELBOW, BODY_LANDMARKS.RIGHT_WRIST),
    (BODY_LANDMARKS.LEFT_WRIST, BODY_LANDMARKS.LEFT_PINKY),
    (BODY_LANDMARKS.RIGHT_WRIST, BODY_LANDMARKS.RIGHT_PINKY),
    (BODY_LANDMARKS.LEFT_WRIST, BODY_LANDMARKS.LEFT_INDEX),
    (BODY_LANDMARKS.RIGHT_WRIST, BODY_LANDMARKS.RIGHT_INDEX),
    (BODY_LANDMARKS.LEFT_WRIST, BODY_LANDMARKS.LEFT_THUMB),
    (BODY_LANDMARKS.RIGHT_WRIST, BODY_LANDMARKS.RIGHT_THUMB),

    # LOWER BODY
    (BODY_LANDMARKS.LEFT_HIP, BODY_LANDMARKS.RIGHT_HIP),
]


# ADD CONNECTION TO MAJOR LOCATIONS
HAND_GRAPH_RELATIONAL = [

    # THUMB TO MAJOR LOCATIONS
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_EYE),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_EYE),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.NOSE),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_SHOULDER),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_SHOULDER),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_ELBOW),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_ELBOW),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_WRIST),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_WRIST),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_PINKY),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_PINKY),
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.LEFT_HIP),   
    (HAND_LANDMARKS.THUMB_TIP, BODY_LANDMARKS.RIGHT_HIP),

    # INDEX FINGER TO MAJOR LOCATIONS
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_EYE),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_EYE),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.NOSE),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_SHOULDER),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_SHOULDER),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_ELBOW),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_ELBOW),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_WRIST),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_WRIST),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_PINKY),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_PINKY),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_HIP),   
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_HIP),

    # MIDDLE FINGER TO MAJOR LOCATIONS
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.LEFT_EYE),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.RIGHT_EYE),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.NOSE),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.LEFT_SHOULDER),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.RIGHT_SHOULDER),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.LEFT_ELBOW),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.RIGHT_ELBOW),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.LEFT_WRIST),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.RIGHT_WRIST),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.LEFT_PINKY),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, BODY_LANDMARKS.RIGHT_PINKY),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.LEFT_HIP),   
    (HAND_LANDMARKS.INDEX_FINGER_TIP, BODY_LANDMARKS.RIGHT_HIP),

    # RING FINGER TO MAJOR LOCATIONS
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_EYE),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_EYE),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.NOSE),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_SHOULDER),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_SHOULDER),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_ELBOW),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_ELBOW),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_WRIST),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_WRIST),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_PINKY),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_PINKY),
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.LEFT_HIP),   
    (HAND_LANDMARKS.RING_FINGER_TIP, BODY_LANDMARKS.RIGHT_HIP),

    # PINKY FINGER TO MAJOR LOCATIONS
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_EYE),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_EYE),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.NOSE),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_SHOULDER),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_SHOULDER),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_ELBOW),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_ELBOW),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_WRIST),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_WRIST),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_PINKY),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_PINKY),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.LEFT_HIP),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, BODY_LANDMARKS.RIGHT_HIP),

]

HAND_2_HAND_RELATIONAL = [
    (HAND_LANDMARKS.THUMB_TIP, HAND_LANDMARKS.THUMB_TIP),
    (HAND_LANDMARKS.INDEX_FINGER_TIP, HAND_LANDMARKS.INDEX_FINGER_TIP),
    (HAND_LANDMARKS.MIDDLE_FINGER_TIP, HAND_LANDMARKS.MIDDLE_FINGER_TIP),
    (HAND_LANDMARKS.RING_FINGER_TIP, HAND_LANDMARKS.RING_FINGER_TIP),
    (HAND_LANDMARKS.PINKY_FINGER_TIP, HAND_LANDMARKS.PINKY_FINGER_TIP),
]