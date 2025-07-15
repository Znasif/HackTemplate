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

class MapIOProcessor(BaseProcessor):
    def __init__(self, enable_sift=True, enable_hands=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Gesture Processor with SIFT model detection and hand tracking
        
        Args:
            model (dict): Model configuration dictionary
            enable_sift (bool): Enable SIFT model detection
            enable_hands (bool): Enable hand detection and tracking
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        super().__init__()
        filename="./models/data.json"
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.model = json.load(f)
                print("Loaded map parameters from file.")
        else:
            print(f"No map parameters file found at {filename}")
            raise FileNotFoundError(f"Could not find model file: {filename}")
        filename="./models/street_map_Market.json"
        
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.market_data = json.load(f)
                print("Loaded map parameters from file.")
            self.pin_matrix = np.zeros((31, 43), dtype=bool)
            self.color_matrix = np.zeros((31, 43, 3), dtype=np.uint8)
            
            # Process hotspots
            for hotspot in self.market_data.get("hotspots", []):
                color = tuple(hotspot.get("color", [255, 0, 0]))
                
                # If positions are specified in the JSON
                if "positions" in hotspot:
                    for i, j in hotspot["positions"]:
                        if 0 <= i < 31 and 0 <= j < 43:
                            self.pin_matrix[i, j] = True
                            self.color_matrix[i, j] = np.array(color)
            
            #____________________________________________________________________>>>>REMOVE THIS LINE TO DRAW MAP
            #self.pin_matrix=None
        else:
            print(f"No map parameters file found at {filename}")
            raise FileNotFoundError(f"Could not find model file: {filename}")
        self.enable_sift = enable_sift
        self.enable_hands = enable_hands
        self.current_desc = ""
        # Dictionary to store pin states
        self.pin_states = [[False for i in range(43)] for j in range(31)]
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
        self.image_map_color = cv2.imread("./models/colorMap.png", cv2.IMREAD_COLOR)
        
        # Homography matrix
        self.H = None
        self.requires_homography = False
        b=5
        self.b=b
        kernel_size = 2 * b + 1
        self.kernel = np.zeros((kernel_size, kernel_size))
        self.kernel[b, b] = 8  # Center weight
        # Neighbor weights at offsets ±b
        for dy in [-b, b]:
            for dx in [-b, b]:
                self.kernel[b + dy, b + dx] = -1
        for dy in [-b, b]:
            self.kernel[b + dy, b] = -1
            self.kernel[b, b + dy] = -1
    
    def initialize_sift_detector(self):
        """Initialize SIFT detector for model recognition"""
        # Load the template image
        self.img_object = cv2.imread(
            "./models/template.png", cv2.IMREAD_GRAYSCALE
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
        self.img_object = cv2.imread("./models/template.png", cv2.IMREAD_GRAYSCALE)
        
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
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}


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
                output = self.render_image_to_pin_display(output)#self.draw_rect_in_image(output, self.image_map_color.shape, self.H)#
        # Initialize detection result
        detection_result = {
            "map_detected": map_detected,
            "zone_description": "",
            "gesture_status": None
        }
        
        # Process hand detection if map detected
        if map_detected and self.enable_hands:
            gesture_loc, gesture_status, output = self.detect_hands(frame, output, self.H)
            
            if gesture_loc is not None:
                # Determine zone from point of interest
                _, zone_desc = self.push_gesture(gesture_loc)
                
                detection_result["zone_description"] = zone_desc
                detection_result["gesture_status"] = gesture_status
                detection_result["gesture_location"] = gesture_loc.tolist()
        
        return output, detection_result["zone_description"]
    
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
                'keypoints0': self.template_data['keypoints'][0].unsqueeze(0),  # Tensor [1, N, 2]
                'scores0': self.template_data['scores'][0].unsqueeze(0),        # Tensor [1, N]
                'descriptors0': self.template_data['descriptors'][0].unsqueeze(0),  # Tensor [1, D, N]
                'keypoints1': frame_data['keypoints'][0].unsqueeze(0),          # Tensor [1, M, 2]
                'scores1': frame_data['scores'][0].unsqueeze(0),                # Tensor [1, M]
                'descriptors1': frame_data['descriptors'][0].unsqueeze(0),       # Tensor [1, D, M]
                'image0': self.template_tensor,  # [1, 1, H, W]
                'image1': frame_tensor      # [1, 1, H, W]
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
        match_confidence = confidence[valid]

        #print(f"SuperGlue: {len(matched_template_kp)} matches")

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
        #print(f"SuperGlue inliers: {total_inliers}")

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
        # If we have already computed the coordinate transform then simply return it
        #if not self.requires_homography and self.H is not None:
        #    return True, self.H, None
            
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
        print(total_inliers)
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
        coors = np.zeros((4, 3), dtype=float)
        
        # Draw the hand annotations on the image
        image_rgb.flags.writeable = True
        #output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
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
        Push gesture position to zone filter
        
        Args:
            position (numpy.ndarray): Position of the gesture
            
        Returns:
            int: Zone ID
        """
        zone_color = self.get_zone(position)
        zone_id, zone_desc = self.get_dict_idx_from_color(zone_color)
        
        self.zone_filter[self.zone_filter_cnt] = zone_id
        self.zone_filter_cnt = (self.zone_filter_cnt + 1) % self.ZONE_FILTER_SIZE
        
        zone = stats.mode(self.zone_filter).mode
        if isinstance(zone, np.ndarray):
            zone = zone[0]
            
        #if np.abs(position[2]) < self.Z_THRESHOLD:
        #    # Find matching zone
        if self.pin_matrix is None:
            return -1, ""
        for zone in self.market_data.get("hotspots", []):
            positions = zone.get("positions", [])
            for x,y in positions:
                if x*43+(y+1)==int(zone_id):
                    print(x, y)
                    zone_desc=zone.get("hotspotTitle", "")+" "+zone.get("hotspotDescription", "")
                    print("HERE---------------->"+zone_desc)
                    if zone_desc!=self.current_desc:
                        self.current_desc = zone_desc
                        return zone, zone_desc
        return -1, ""

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
    
    def get_dict_idx_from_color(self, color):
        """
        Get zone ID from color
        
        Args:
            color (tuple): Color (B, G, R)
            
        Returns:
            int: Zone ID
        """
        # Get zone mappings from model
        zones = self.model['hotspots']
        
        # Find matching zone
        for zone in zones:
            zone_color = zone.get("color", None)
            if zone_color and tuple(zone_color) == color:
                return int(zone["hotspotTitle"]), zone["hotspotDescription"]
        
        return -1, ""
    
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
        # Create a copy of the input image
        output = image.copy()
        
        # Get the map image and ensure it has the same number of channels as the output
        map_image = self.image_map_color.copy()
        if len(map_image.shape) == 2 and len(output.shape) == 3:
            map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
        
        # Create a mask for the map area in the output image
        h, w = map_image.shape[:2]
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Create coordinates for the four corners of the map in the map's coordinate system
        map_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        map_corners = map_corners.reshape(-1, 1, 2)
        
        # Transform the map corners to the camera view using inverse homography
        H_inv = np.linalg.inv(H)
        camera_corners = cv2.perspectiveTransform(map_corners, H_inv)
        
        # Fill the polygon area in the mask
        camera_corners_int = np.int32(camera_corners.reshape(-1, 2))
        cv2.fillPoly(mask, [camera_corners_int], 255)
        
        # Warp the map image to fit the perspective in the camera view
        h_image, w_image = image.shape[:2]
        warped_map = cv2.warpPerspective(map_image, H_inv, (w_image, h_image))
        
        # Create the overlay with half opacity
        # Only apply the overlay where the mask is non-zero
        alpha = 0.5  # Opacity level
        for c in range(3):  # For each color channel
            output[:, :, c] = np.where(
                mask > 0,
                output[:, :, c] * (1 - alpha) + warped_map[:, :, c] * alpha,
                output[:, :, c]
            )
        
        # Optional: Draw the outline to show the boundaries clearly
        cv2.polylines(output, [camera_corners_int], True, (0, 255, 0), 2)
        if(self.detect_pin_states(image, H)):
            text_repr = "\n".join([''.join(['●' if pixel else '○' for pixel in row]) for row in self.pin_states])
            #print(text_repr, end="\r")
            return self.vis_image
        return output

    def detect_pin_states(self, image, H):
        """
        Detect whether pins are up (reflective) or down at hotspot locations
        using the pixel-based convolution approach adapted for ROIs
        """
        # Create a copy of the input image for visualization
        self.vis_image = image.copy()
        
        # Convert input image to grayscale for brightness analysis
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get hotspots from model
        hotspots = self.model.get("hotspots", [])
        if not hotspots:
            return False
        
        # Constants for analysis
        PIN_RADIUS = 5  # Radius around pin center to analyze
        b = self.b  # Neighborhood size for convolution (from pixel-based code)
        
        # Extract all positions, rows, and columns at once
        positions = []
        rows = []
        cols = []
        
        for idx, hotspot in enumerate(hotspots):
            if "position" in hotspot:
                positions.append(hotspot["position"])
                count = int(hotspot["hotspotTitle"])
                row, col = int((count-1)/43), (count-1)%43
                rows.append(row)
                cols.append(col)
        
        # Convert to numpy arrays
        positions = np.array(positions, dtype=np.float32)
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        
        if positions.size == 0:
            return False
        
        # Transform all positions to camera view at once
        positions_map = np.array([positions], dtype=np.float32)
        positions_camera = cv2.perspectiveTransform(positions_map, np.linalg.inv(H))
        positions_camera = positions_camera[0]
        
        # Convert to integers and filter out points outside the image
        cx_values = positions_camera[:, 0].astype(np.int32)
        cy_values = positions_camera[:, 1].astype(np.int32)
        
        # Create masks for valid points
        valid_mask = (cx_values >= b) & (cx_values < image.shape[1] - b) & \
                    (cy_values >= b) & (cy_values < image.shape[0] - b)
        
        # Filter to keep only valid points
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            print("False")
            return False
        
        # Filter arrays to keep only valid points
        cx_values = cx_values[valid_mask]
        cy_values = cy_values[valid_mask]
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        
        # Initialize pin state classifications
        average_response = np.zeros(len(cx_values), dtype=np.float32)
        
        # Process each hotspot's ROI using the pixel-based convolution logic
        for i, (cx, cy, row, col) in enumerate(zip(cx_values, cy_values, rows, cols)):
            # Extract ROI (ensure it's large enough to include the neighborhood)
            x_min = max(0, cx - PIN_RADIUS - b)
            x_max = min(image.shape[1], cx + PIN_RADIUS + b)
            y_min = max(0, cy - PIN_RADIUS - b)
            y_max = min(image.shape[0], cy + PIN_RADIUS + b)
            
            # Get the ROI from the grayscale image
            roi = gray_image[y_min:y_max, x_min:x_max]
            
            # Apply the convolution operation at the center of the ROI
            roi_h, roi_w = roi.shape
            center_x = cx - x_min
            center_y = cy - y_min
            
            # Ensure the center is within bounds for the convolution
            if (center_x >= b and center_x < roi_w - b and 
                center_y >= b and center_y < roi_h - b):
                feature_map = signal.convolve2d(roi, self.kernel, mode='valid')
                # Extract a small region around the center for averaging
                # Adjust for 'valid' mode (reduces size by kernel_size-1)
                fm_y = center_y - b
                fm_x = center_x - b
                # Define a 3x3 region around the center (adjust size as needed)
                region_size = PIN_RADIUS-1
                half_size = region_size // 2
                region_y_min = max(0, fm_y - half_size)
                region_y_max = min(feature_map.shape[0], fm_y + half_size + 1)
                region_x_min = max(0, fm_x - half_size)
                region_x_max = min(feature_map.shape[1], fm_x + half_size + 1)
                
                # Compute the average value in the region
                if (region_y_max > region_y_min and region_x_max > region_x_min):
                    region = feature_map[region_y_min:region_y_max, region_x_min:region_x_max]
                    average_response[i] = np.mean(region)
                else:
                    average_response[i] = 0  # Fallback if region is invalid
        #write code to do k=2 knn based clustering on all the average_response values, it is possible that all the pins may be down in which case the all may belong to down cluster, but it 
        #is highly unlikely to all pins be up, so cluster that way
        classifications = np.zeros(len(cx_values), dtype=np.int32)
        # Apply k-means clustering with k=2 to classify pins as up or down
        if len(average_response) > 0:
            # Reshape for sklearn
            X = average_response.reshape(-1, 1)
            
            # Apply KMeans with k=2
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            cluster_centers = kmeans.cluster_centers_
            
            # Determine which cluster represents "up" pins (higher response values)
            up_cluster = 0 if cluster_centers[0] > cluster_centers[1] else 1
            
            # Classify each pin based on cluster assignment
            classifications = (kmeans.labels_ == up_cluster).astype(np.int32)
            print(np.max(average_response))
            # Special case: If all pins might be down, check the separation between clusters
            # if np.max(average_response) < 1:  # Define appropriate threshold
            #     classifications[:] = 0  # All pins are down
        if np.any(classifications):
            for i, (cx, cy) in enumerate(zip(cx_values, cy_values)):
                if average_response[i]>=0:
                    #self.pin_states[]
                    cv2.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 255, 0), -1)  # Green for up
                else:
                    cv2.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 0, 255), -1)  # Red for down
            return True
        return False
    
    def render_image_to_pin_display(self, frame):
        """
        Renders an image to the pin display and creates a visualization using homography.
        
        Args:
            display_image (numpy.ndarray): Input image to be displayed on the pin matrix
            frame (numpy.ndarray): Camera frame to draw the visualization on
            
        Returns:
            tuple: (pin_matrix, visualization)
                - pin_matrix (numpy.ndarray): Binary 2D array representing pin states
                - visualization (numpy.ndarray): Visualization image with pin matrix projected onto frame
        """
        # Create visualization using homography
        visualization = self.visualize_pin_matrix_with_homography(frame)
        
        return visualization

    def visualize_pin_matrix_with_homography(self, frame):
        """
        Visualizes the pin matrix by using homography to project onto the image_map_color
        and drawing circle indicators for each pin.
        
        Args:
            pin_matrix (numpy.ndarray): Binary 2D array representing pin states (31x43)
            frame (numpy.ndarray): Camera frame to draw on
            
        Returns:
            numpy.ndarray: Visualization with the pin matrix projected onto the frame
        """
        # Create a copy of the input frame
        output = frame.copy()
        
        # Check if homography matrix exists
        if self.H is None:
            return output
        
        # Create a copy of the map image
        map_image = self.image_map_color.copy()
        map_image[:] = (0, 0, 0)

        # Create a visualization of the pin matrix on the map
        for hotspot in self.model.get("hotspots", []):
            if "position" in hotspot and "hotspotTitle" in hotspot:
                # Get position from hotspot
                x, y = int(hotspot["position"][0]), int(hotspot["position"][1])
                
                # Get row and column from hotspot title
                count = int(hotspot["hotspotTitle"])
                row, col = int((count-1)/43), (count-1) % 43
                
                # Check if this position is within bounds
                if 0 <= row < self.pin_matrix.shape[0] and 0 <= col < self.pin_matrix.shape[1]:
                    # Determine color based on pin state
                    if self.pin_matrix[row, col]:
                        color = (int(self.color_matrix[row, col][2]), 
                                int(self.color_matrix[row, col][1]), 
                                int(self.color_matrix[row, col][0]))
                    else:
                        color = (0, 0, 0)
                    # Draw the circle on the map image
                    cv2.circle(map_image, (x, y), 5, color, -1)  # Filled circle
        
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
        black_pixels_mask = (warped_map[:,:,0] == 0) & (warped_map[:,:,1] == 0) & (warped_map[:,:,2] == 0)
        # Alpha blend the warped map with the output using different alpha values
        # Use alpha = 0.5 for normal pixels, and alpha = 0.5 for black pixels too
        for c in range(3):  # For each color channel
            # Apply different alpha blending based on pixel color
            output[:, :, c] = np.where(
                (mask > 0) & ~black_pixels_mask,  # Non-black pixels within mask
                output[:, :, c] * 0.5 + warped_map[:, :, c] * 0.5,  # Regular 0.5 alpha
                np.where(
                    (mask > 0) & black_pixels_mask,  # Black pixels within mask
                    output[:, :, c] * 0.5 + warped_map[:, :, c] * 0.5,  # Also 0.5 alpha for black pixels
                    output[:, :, c]  # Outside mask, keep original
                )
            )
        
        # Optional: Draw the outline to show the boundaries clearly
        cv2.polylines(output, [camera_corners_int], True, (0, 255, 0), 2)
        
        return output

    def reset_homography(self):
        """Reset homography to force recalculation"""
        self.requires_homography = True
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        #if self.enable_hands:
        #    self.hands.close()
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

processor = MapIOProcessor()
app = processor.app