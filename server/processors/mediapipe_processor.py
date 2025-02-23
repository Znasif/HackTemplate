from .base_processor import BaseProcessor
import mediapipe as mp
import numpy as np
import cv2

class MediaPipeProcessor(BaseProcessor):
    def __init__(self, 
                 enable_face=True, 
                 enable_pose=True, 
                 enable_hands=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe processor with specified detectors
        
        Args:
            enable_face (bool): Enable face detection and landmarks
            enable_pose (bool): Enable pose detection and landmarks
            enable_hands (bool): Enable hand detection and landmarks
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        super().__init__()
        
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face detection if enabled
        if self.enable_face:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
        # Initialize pose detection if enabled
        if self.enable_pose:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
        # Initialize hand detection if enabled
        if self.enable_hands:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                model_complexity=1,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    def process_frame(self, frame):
        """
        Process frame using MediaPipe to detect and draw landmarks
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame (numpy.ndarray): Frame with drawn landmarks
                - detections (dict): Dictionary containing detection results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create output frame as copy of input
        output = frame.copy()
        
        # Initialize detections dictionary
        detections = {
            'face_landmarks': None,
            'pose_landmarks': None,
            'hand_landmarks': None
        }
        
        # Process face detection
        if self.enable_face:
            face_results = self.face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=output,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                detections['face_landmarks'] = face_results.multi_face_landmarks
        
        # Process pose detection
        if self.enable_pose:
            pose_results = self.pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                detections['pose_landmarks'] = pose_results.pose_landmarks
        
        # Process hand detection
        if self.enable_hands:
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                detections['hand_landmarks'] = hand_results.multi_hand_landmarks
        
        return output, ""

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

    def __del__(self):
        """
        Clean up MediaPipe resources
        """
        if self.enable_face:
            self.face_mesh.close()
        if self.enable_pose:
            self.pose.close()
        if self.enable_hands:
            self.hands.close()