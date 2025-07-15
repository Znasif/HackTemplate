import cv2
import numpy as np
from typing import Dict, Union, Tuple, List, Optional
from .base_processor import BaseProcessor
import mediapipe as mp

class FingerCounterProcessor(BaseProcessor):
    def __init__(self, 
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75):
        """
        Initialize Finger Counter processor with hand detection
        
        Args:
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        super().__init__()
        
        # Initialize MediaPipe hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Finger tip landmark IDs (thumb, index, middle, ring, pinky)
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame to count fingers on right hand and draw results
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame (numpy.ndarray): Frame with drawn landmarks and finger count
                - detections (dict): Dictionary containing finger count and hand landmarks
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = frame.copy()
        
        # Initialize detections dictionary
        detections = {
            'hand_landmarks': None,
            'finger_count': 0,
            'hand_type': None
        }

        detections['finger_count'] = 0
        
        # Process hand detection
        hand_results = self.hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Check if it's the right hand
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                if hand_label == 'Right':
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Get landmark coordinates
                    coords = self.get_landmark_coordinates(hand_landmarks, output.shape)
                    
                    # Count fingers
                    fingers = []
                    
                    # Thumb (comparing x-coordinate)
                    if coords[self.tip_ids[0]][0] > coords[self.tip_ids[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    
                    # Other fingers (comparing y-coordinate)
                    for id in range(1, 5):
                        if coords[self.tip_ids[id]][1] < coords[self.tip_ids[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    
                    # Calculate total fingers
                    total_fingers = fingers.count(1)
                    
                    # Draw finger count
                    cv2.rectangle(output, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
                    cv2.putText(output, str(total_fingers), (45, 375), 
                              cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                    
                    detections['hand_landmarks'] = hand_landmarks
                    detections['finger_count'] = total_fingers
                    detections['hand_type'] = hand_label
        
        
        return output, str(detections['finger_count'])

    def get_landmark_coordinates(self, landmarks, image_shape):
        """
        Convert normalized landmarks to pixel coordinates
        
        Args:
            landmarks: MediaPipe landmark object
            image_shape: Shape of the image (height, width)
            
        Returns:
            list: List of (x, y) coordinates
        """
        height, width = image_shape[:2]
        coords = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coords.append((x, y))
        return coords

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Process point cloud data (not implemented for finger counting)
        
        Args:
            point_cloud_data (Dict): Input point cloud data
            
        Returns:
            Tuple containing input data and message
        """
        return point_cloud_data, {"message": "FingerCounterProcessor does not process point cloud data."}

    def __del__(self):
        """
        Clean up MediaPipe resources
        """
        self.hands.close()

processor = FingerCounterProcessor()
app = processor.app