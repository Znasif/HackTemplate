import cv2
import numpy as np
import mediapipe as mp
from scipy import stats, signal
from collections import deque
import time, os, json
import torch
from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.superglue import SuperGlue
from google.protobuf.json_format import MessageToDict
from typing import Dict, Union, Tuple, List, Optional
from .base_processor import BaseProcessor

class RegionIOProcessor(BaseProcessor):
    def __init__(self, enable_sift=True, enable_hands=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5, folder_path = "./models/cars/panel3_output/"):
        """
        Initialize MediaPipe Gesture Processor with SIFT model detection and hand tracking
        
        Args:
            enable_sift (bool): Enable SIFT model detection
            enable_hands (bool): Enable hand detection and tracking
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        super().__init__()
        
        # Load regions configuration
        self.folder_path = folder_path
        filename = self.folder_path + "regions.json"
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.regions_data = json.load(f)
                print("Loaded regions configuration from file.")
        else:
            print(f"No regions file found at {filename}")
            raise FileNotFoundError(f"Could not find regions file: {filename}")
        
        self.enable_sift = enable_sift
        self.enable_hands = enable_hands
        self.current_desc = ""
        
        # Dictionary to store region states (detection_id -> state)
        self.region_states = {region['detection_id']: False 
                            for region in self.regions_data.get('regions', [])}
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        if self.enable_hands:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        # Initialize SIFT model detector
        if self.enable_sift:
            self.initialize_sift_detector()
        else:
            self.initialize_HLoc_detector()
            
        # Initialize zone interaction
        self.initialize_interaction_policy()
        
        # Initialize movement filters
        self.movement_filter = self.MovementMedianFilter()
        self.gesture_detector = self.GestureDetector()
        
        # Load zone mapping image
        filename = self.folder_path + "colorMap.png"
        self.image_map_color = cv2.imread(filename, cv2.IMREAD_COLOR)
        
        # Homography matrix
        self.H = None
        self.requires_homography = False
    
    def initialize_sift_detector(self):
        """Initialize SIFT detector for model recognition"""
        # Load the template image
        filename = self.folder_path + "template.jpg"
        self.img_object = cv2.imread(
            filename, cv2.IMREAD_GRAYSCALE
        )

        # Detect SIFT keypoints
        self.sift_detector = cv2.SIFT_create()
        self.keypoints_obj, self.descriptors_obj = self.sift_detector.detectAndCompute(
            self.img_object, mask=None
        )
        self.MIN_INLIER_COUNT = 25
    
    def initialize_HLoc_detector(self):
        """Initialize SuperPoint detector and SuperGlue matcher from HLoc"""
        
        # Initialize SuperPoint feature extractor
        self.superpoint = SuperPoint({
            'max_keypoints': 4096,
            'keypoint_threshold': 0.005,
            'remove_borders': 4,
            'nms_radius': 4,
        }).eval()
        
        # Initialize SuperGlue matcher
        self.superglue = SuperGlue({
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }).eval()
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.superpoint = self.superpoint.to(self.device)
        self.superglue = self.superglue.to(self.device)
        
        # Load the template image
        filename = self.folder_path + "template.jpg"
        self.img_object = cv2.imread(
            filename, cv2.IMREAD_GRAYSCALE
        )
        # Process template image for SuperPoint
        self.template_tensor = torch.from_numpy(self.img_object).float() / 255.
        self.template_tensor = self.template_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Extract template features
        with torch.no_grad():
            self.template_data = self.superpoint({'image': self.template_tensor})
        
        # Minimum number of inliers for a valid detection
        self.MIN_INLIER_COUNT = 15

    def initialize_interaction_policy(self):
        """Initialize zone interaction policy"""
        self.ZONE_FILTER_SIZE = 10
        self.Z_THRESHOLD = 2.0
        self.zone_filter = -1 * np.ones(self.ZONE_FILTER_SIZE, dtype=int)
        self.zone_filter_cnt = 0
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        return point_cloud_data, {"message": "RegionIOProcessor received point cloud data but does not process it further."}

    def process_frame(self, frame):
        """
        Process frame using MediaPipe and SIFT to detect models and hand gestures
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, detections_json)
                - processed_frame (numpy.ndarray): Frame with drawn landmarks
                - detections_json (str): JSON string with detection results
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Convert to grayscale for SIFT processing
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process SIFT detection if enabled
        map_detected = False
        if self.enable_sift:
            map_detected, self.H, _ = self.detect_model(frame_gray)
            if map_detected:
                output = self.draw_rect_in_image(output, self.image_map_color.shape, self.H)
        else:
            map_detected, self.H, _ = self.detect_model_with_HLoc(frame_gray)
            if map_detected:
                output = self.render_regions_overlay(output)
        
        # Initialize detection result
        detection_result = {
            "map_detected": map_detected,
            "region_description": "",
            "gesture_status": None
        }
        
        # Process hand detection if map detected
        if map_detected and self.enable_hands:
            gesture_loc, gesture_status, output = self.detect_hands(frame, output, self.H)
            
            if gesture_loc is not None:
                # Determine region from point of interest
                region_info = self.push_gesture(gesture_loc)
                
                detection_result["region_description"] = region_info
                detection_result["gesture_status"] = gesture_status
                detection_result["gesture_location"] = gesture_loc.tolist()
        
        print(detection_result["region_description"])
        return output, detection_result["region_description"]
    
    def detect_model_with_HLoc(self, frame_gray):
        """
        Detect model in frame using SuperPoint features and SuperGlue matcher
        Args:
            frame_gray (numpy.ndarray): Grayscale input frame
        Returns:
            tuple: (success, homography, transform_vector)
        """
        # If we have already computed the homography and don't require recomputation, return it
        if not self.requires_homography and self.H is not None:
            return True, self.H, None

        # Preprocess frame for SuperPoint
        frame_tensor = torch.from_numpy(frame_gray).float() / 255.
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Extract features from current frame
        with torch.no_grad():
            frame_data = self.superpoint({'image': frame_tensor})

            # Create SuperGlue input dict with required keys
            superglue_input = {
                'keypoints0': self.template_data['keypoints'][0].unsqueeze(0),
                'scores0': self.template_data['scores'][0].unsqueeze(0),
                'descriptors0': self.template_data['descriptors'][0].unsqueeze(0),
                'keypoints1': frame_data['keypoints'][0].unsqueeze(0),
                'scores1': frame_data['scores'][0].unsqueeze(0),
                'descriptors1': frame_data['descriptors'][0].unsqueeze(0),
                'image0': self.template_tensor,
                'image1': frame_tensor
            }

            # Match features using SuperGlue
            matches_data = self.superglue(superglue_input)

        # Extract matches
        matches = matches_data['matches0'][0].cpu().numpy()
        confidence = matches_data['matching_scores0'][0].cpu().numpy()

        # Filter valid matches
        valid = matches > -1

        # Get matched keypoints
        template_kp = self.template_data['keypoints'][0].cpu().numpy()
        frame_kp = frame_data['keypoints'][0].cpu().numpy()

        # Get matched pairs
        matched_template_kp = template_kp[valid]
        matched_frame_kp = frame_kp[matches[valid]]

        # Need at least 4 good matches to compute homography
        if len(matched_template_kp) < 4:
            return False, None, None

        # Compute homography
        H, mask = cv2.findHomography(
            matched_frame_kp, matched_template_kp,
            cv2.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )

        # Count inliers
        total_inliers = np.sum(mask) if mask is not None else 0

        # Check if we have enough inliers
        if total_inliers > self.MIN_INLIER_COUNT:
            self.H = H
            return True, H, None
        elif self.H is not None:
            return True, self.H, None
        else:
            return False, None, None

    def detect_model(self, frame_gray):
        """
        Detect model in frame using SIFT
        
        Args:
            frame_gray (numpy.ndarray): Grayscale input frame
            
        Returns:
            tuple: (success, homography, transform_vector)
        """
        keypoints_scene, descriptors_scene = self.sift_detector.detectAndCompute(frame_gray, None)
        
        # If no descriptors found, return failure
        if descriptors_scene is None or len(descriptors_scene) < 4:
            return False, None, None
            
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.descriptors_obj, descriptors_scene, 2)

        # Only keep uniquely good matches
        RATIO_THRESH = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < RATIO_THRESH * n.distance:
                good_matches.append(m)
                
        # Need at least 4 good matches to compute homography
        if len(good_matches) < 4:
            return False, None, None
            
        # Extract matched keypoints
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # Get the keypoints from the good matches
            obj[i, 0] = self.keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
            
        # Compute homography and find inliers
        H, mask_out = cv2.findHomography(
            scene, obj, cv2.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )
        
        # Count inliers
        total_inliers = sum([int(i) for i in mask_out])
        
        # Check if we have enough inliers
        if total_inliers > self.MIN_INLIER_COUNT:
            self.H = H
            self.requires_homography = False
            return True, H, None
        elif self.H is not None:
            return True, self.H, None
        else:
            return False, None, None
    
    def detect_hands(self, image, output_image, H):
        """
        Detect hand landmarks and gestures
        
        Args:
            image (numpy.ndarray): Input image
            H (numpy.ndarray): Homography matrix
            
        Returns:
            tuple: (index_position, movement_status, output_image)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        handedness = list()
        results = self.hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        index_pos = None
        movement_status = None
        
        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left/right)
                if results.multi_handedness:
                    handedness.append(MessageToDict(results.multi_handedness[h])['classification'][0]['label'])
                
                # Calculate ratios for each finger
                finger_ratios = self._calculate_finger_ratios(hand_landmarks)
                
                # Get index finger position
                position = np.matmul(H, np.array([
                    hand_landmarks.landmark[8].x * image.shape[1],
                    hand_landmarks.landmark[8].y * image.shape[0],
                    1
                ]))

                # Initialize index position
                if index_pos is None:
                    index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)

                # Check if index finger is pointing (extended while others are closed)
                is_pointing = (finger_ratios[1] > 0.7 and  # Index finger extended
                            finger_ratios[2] < 0.95 and  # Middle finger closed
                            finger_ratios[3] < 0.95 and  # Ring finger closed
                            finger_ratios[4] < 0.95)    # Little finger closed

                if is_pointing:
                    # Handle multiple hands or same hand pointing
                    if movement_status != "pointing" or (len(handedness) > 1 and handedness[1] == handedness[0]):
                        index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)
                        movement_status = "pointing"
                    else:
                        index_pos = np.append(index_pos,
                                            np.array([position[0] / position[2], position[1] / position[2], 0],
                                                    dtype=float))
                        movement_status = "too_many"

                    H_inv = np.linalg.inv(H)
                    # Directly use the de-homogenized 'position' to draw the circle
                    index_tip_homogeneous = np.array([index_pos[-3], index_pos[-2], 1])
                    transformed_tip = np.dot(H_inv, index_tip_homogeneous)
                    transformed_x, transformed_y = int(transformed_tip[0] / transformed_tip[2]), int(transformed_tip[1] / transformed_tip[2])

                    # Draw the white circle at the transformed position
                    cv2.circle(output_image, (transformed_x, transformed_y), 10, (255, 255, 255), -1)
                elif movement_status != "pointing":
                    movement_status = "moving"
                    # Draw all hand landmarks if not pointing
                    self.mp_drawing.draw_landmarks(
                        output_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        
        return index_pos, movement_status, output_image
    
    def _calculate_finger_ratios(self, hand_landmarks):
        """
        Calculate ratios for each finger to determine if they're extended
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            list: Ratios for each finger [thumb, index, middle, ring, little]
        """
        finger_ratios = []
        
        # Finger landmark indices
        finger_indices = [
            [1, 2, 3, 4],     # Thumb
            [5, 6, 7, 8],     # Index
            [9, 10, 11, 12],  # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20]  # Little
        ]
        
        # Calculate ratio for each finger
        for finger in finger_indices:
            coors = np.zeros((4, 3), dtype=float)
            for i, k in enumerate(finger):
                coors[i, 0] = hand_landmarks.landmark[k].x
                coors[i, 1] = hand_landmarks.landmark[k].y
                coors[i, 2] = hand_landmarks.landmark[k].z
            
            ratio = self._calculate_ratio(coors)
            finger_ratios.append(ratio)
        
        return finger_ratios
    
    def _calculate_ratio(self, coors):
        """
        Calculate ratio to determine if finger is extended
        Ratio is 1 if points are collinear, lower otherwise (minimum is 0)
        
        Args:
            coors (numpy.ndarray): Coordinates of finger joints
            
        Returns:
            float: Ratio indicating if finger is extended
        """
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])

        return d / (a + b + c)
    
    def push_gesture(self, position):
        """
        Push gesture position to zone filter and determine region
        
        Args:
            position (numpy.ndarray): Position of the gesture
            
        Returns:
            str: Region description
        """
        zone_color = self.get_zone(position)
        detection_id, class_name = self.get_region_from_color(zone_color)
        
        if detection_id:
            region_desc = f"{class_name}"
            print(region_desc)
            if region_desc != self.current_desc:
                self.current_desc = region_desc
                return region_desc
        
        return ""

    def get_zone(self, position):
        """
        Get zone color at position
        
        Args:
            position (numpy.ndarray): Position to check
            
        Returns:
            tuple: Color (B, G, R) at position
        """
        x, y = int(position[0]), int(position[1])
        h, w = self.image_map_color.shape[:2]
        
        # Check if position is within image bounds
        if 0 <= x < w and 0 <= y < h:
            return tuple(self.image_map_color[y, x])
        else:
            return (0, 0, 0)
    
    def get_region_from_color(self, color):
        """
        Get region info from color
        
        Args:
            color (tuple): Color (B, G, R)
            
        Returns:
            tuple: (detection_id, class_name)
        """
        # Get regions from loaded data
        regions = self.regions_data.get('regions', [])
        
        # Find matching region by color
        for region in regions:
            region_color = region.get("color", None)
            if region_color and tuple(region_color) == color:
                return region["detection_id"], region["class"]
        
        return None, None
    
    def draw_rect_in_image(self, image, sz, H):
        """
        Overlay the map image with half opacity in the camera view
        
        Args:
            image (numpy.ndarray): Input image (camera frame)
            sz (tuple): Size of rectangle (height, width) - from the map image
            H (numpy.ndarray): Homography matrix
            
        Returns:
            numpy.ndarray: Image with map overlay
        """
        return self.render_regions_overlay(image)

    def render_regions_overlay(self, frame):
        """
        Renders regions overlay using homography projection
        
        Args:
            frame (numpy.ndarray): Camera frame to draw on
            
        Returns:
            numpy.ndarray: Frame with regions overlay
        """
        # Create a copy of the input frame
        output = frame.copy()
        
        # Check if homography matrix exists
        if self.H is None:
            return output
        
        # Create a copy of the color map
        map_image = self.image_map_color.copy()
        
        # Create a mask for the map area in the output image
        h, w = map_image.shape[:2]
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Create coordinates for the four corners of the map
        map_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        map_corners = map_corners.reshape(-1, 1, 2)
        
        # Transform the map corners to the camera view using inverse homography
        H_inv = np.linalg.inv(self.H)
        camera_corners = cv2.perspectiveTransform(map_corners, H_inv)
        
        # Fill the polygon area in the mask
        camera_corners_int = np.int32(camera_corners.reshape(-1, 2))
        cv2.fillPoly(mask, [camera_corners_int], 255)
        
        # Warp the map image to fit the perspective in the camera view
        h_frame, w_frame = frame.shape[:2]
        warped_map = cv2.warpPerspective(map_image, H_inv, (w_frame, h_frame))
        
        # Alpha blend the warped map with the output
        alpha = 0.5  # Opacity level
        for c in range(3):  # For each color channel
            output[:, :, c] = np.where(
                mask > 0,
                output[:, :, c] * (1 - alpha) + warped_map[:, :, c] * alpha,
                output[:, :, c]
            )
        
        # Optional: Draw the outline to show the boundaries clearly
        cv2.polylines(output, [camera_corners_int], True, (0, 255, 0), 2)
        
        return output

    def reset_homography(self):
        """Reset homography to force recalculation"""
        self.requires_homography = True
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        pass
    
    class MovementMedianFilter:
        """Filter for smoothing movement using median filtering"""
        def __init__(self):
            self.MAX_QUEUE_LENGTH = 30
            self.positions = deque(maxlen=30)
            self.times = deque(maxlen=30)
            self.AVERAGING_TIME = 0.7

        def push_position(self, position):
            self.positions.append(position)
            now = time.time()
            self.times.append(now)
            i = len(self.times)-1
            Xs = []
            Ys = []
            Zs = []
            while i >= 0 and now - self.times[i] < self.AVERAGING_TIME:
                Xs.append(self.positions[i][0])
                Ys.append(self.positions[i][1])
                Zs.append(self.positions[i][2])
                i -= 1
            return np.array([np.median(Xs), np.median(Ys), np.median(Zs)])
    
    class GestureDetector:
        """Detector for gesture movements and stillness"""
        def __init__(self):
            self.MAX_QUEUE_LENGTH = 30
            self.positions = deque(maxlen=30)
            self.times = deque(maxlen=30)
            self.DWELL_TIME_THRESH = 0.75
            self.X_MVMNT_THRESH = 0.95
            self.Y_MVMNT_THRESH = 0.95
            self.Z_MVMNT_THRESH = 4.0

        def push_position(self, position):
            self.positions.append(position)
            now = time.time()
            self.times.append(now)
            i = len(self.times)-1
            Xs = []
            Ys = []
            Zs = []
            while (i >= 0 and now - self.times[i] < self.DWELL_TIME_THRESH):
                Xs.append(self.positions[i][0])
                Ys.append(self.positions[i][1])
                Zs.append(self.positions[i][2])
                i -= 1
            
            if len(Xs) > 0:
                Xdiff = max(Xs) - min(Xs)
                Ydiff = max(Ys) - min(Ys)
                Zdiff = max(Zs) - min(Zs)
                
                if Xdiff < self.X_MVMNT_THRESH and Ydiff < self.Y_MVMNT_THRESH and Zdiff < self.Z_MVMNT_THRESH:
                    return np.array([sum(Xs)/float(len(Xs)), sum(Ys)/float(len(Ys)), sum(Zs)/float(len(Zs))]), 'still'
            
            return position, 'moving'

