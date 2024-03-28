import cv2
import mediapipe as mp
import numpy as np
from ..config import *


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def get_pose_estimation(video_path):

    cap = cv2.VideoCapture(video_path)

    landmarks = {
        'pose': {
            'NOSE': [ ],
            'LEFT_EYE_INNER': [ ],
            'LEFT_EYE': [ ],
            'LEFT_EYE_OUTER': [ ],
            'RIGHT_EYE_INNER': [ ],
            'RIGHT_EYE': [ ],
            'RIGHT_EYE_OUTER': [ ],
            'LEFT_EAR': [ ],
            'RIGHT_EAR': [ ],
            'MOUTH_LEFT': [ ],
            'MOUTH_RIGHT': [ ],
            'LEFT_SHOULDER': [ ],
            'RIGHT_SHOULDER': [ ],
            'LEFT_ELBOW': [ ],
            'RIGHT_ELBOW': [ ],
            'LEFT_WRIST': [ ],
            'RIGHT_WRIST': [ ],
            'LEFT_PINKY': [ ],
            'RIGHT_PINKY': [ ],
            'LEFT_INDEX': [ ],
            'RIGHT_INDEX': [ ],
            'LEFT_THUMB': [ ],
            'RIGHT_THUMB': [ ],
            'LEFT_HIP': [ ],
            'RIGHT_HIP': [ ],
            'LEFT_KNEE': [ ],
            'RIGHT_KNEE': [ ],
            'LEFT_ANKLE': [ ],
            'RIGHT_ANKLE': [ ],
            'LEFT_HEEL': [ ],
            'RIGHT_HEEL': [ ],
            'LEFT_FOOT_INDEX': [ ],
            'RIGHT_FOOT_INDEX': [ ],
        },
        'right': {
            'WRIST': [ ],
            'THUMB_CMC': [ ],
            'THUMB_MCP': [ ],
            'THUMB_IP': [ ],
            'THUMB_TIP': [ ],
            'INDEX_FINGER_MCP': [ ],
            'INDEX_FINGER_PIP': [ ],
            'INDEX_FINGER_DIP': [ ],
            'INDEX_FINGER_TIP': [ ],
            'MIDDLE_FINGER_MCP': [ ],
            'MIDDLE_FINGER_PIP': [ ],
            'MIDDLE_FINGER_DIP': [ ],
            'MIDDLE_FINGER_TIP': [ ],
            'RING_FINGER_MCP': [ ],
            'RING_FINGER_PIP': [ ],
            'RING_FINGER_DIP': [ ],
            'RING_FINGER_TIP': [ ],
            'PINKY_MCP': [ ],
            'PINKY_PIP': [ ],
            'PINKY_DIP': [ ],
            'PINKY_TIP': [ ]
        },
        'left': {
            'WRIST': [ ],
            'THUMB_CMC': [ ],
            'THUMB_MCP': [ ],
            'THUMB_IP': [ ],
            'THUMB_TIP': [ ],
            'INDEX_FINGER_MCP': [ ],
            'INDEX_FINGER_PIP': [ ],
            'INDEX_FINGER_DIP': [ ],
            'INDEX_FINGER_TIP': [ ],
            'MIDDLE_FINGER_MCP': [ ],
            'MIDDLE_FINGER_PIP': [ ],
            'MIDDLE_FINGER_DIP': [ ],
            'MIDDLE_FINGER_TIP': [ ],
            'RING_FINGER_MCP': [ ],
            'RING_FINGER_PIP': [ ],
            'RING_FINGER_DIP': [ ],
            'RING_FINGER_TIP': [ ],
            'PINKY_MCP': [ ],
            'PINKY_PIP': [ ],
            'PINKY_DIP': [ ],
            'PINKY_TIP': [ ]
        }
    }

    # NOTE: There can be undetectable landmarks, check out later
    with mp_holistic.Holistic(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE, 
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                # Append pose landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['pose'].keys()), results.pose_landmarks.landmark):
                    landmarks['pose'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ]))
            else:
                for landmark_name in landmarks['pose'].keys():
                    landmarks['pose'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                        None,
                    ]))

            if results.right_hand_landmarks:
                # Append right hand landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['right'].keys()), results.right_hand_landmarks.landmark):
                    landmarks['right'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ]))
            else:
                for landmark_name in landmarks['right'].keys():
                    landmarks['right'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                        None,
                    ]))

            if results.left_hand_landmarks:
                # Append left hand landmarks to the list
                for landmark_name, landmark in zip(list(landmarks['left'].keys()), results.left_hand_landmarks.landmark):
                    landmarks['left'][landmark_name].append(np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ]))
            else:
                for landmark_name in landmarks['left'].keys():
                    landmarks['left'][landmark_name].append(np.array([
                        None,
                        None,
                        None,
                        None,
                    ]))

    cap.release()

    return landmarks