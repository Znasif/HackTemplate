import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy import stats
from collections import deque
import time, os, json
import torch
from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.superglue import SuperGlue
from google.protobuf.json_format import MessageToDict
from .base_processor import BaseProcessor

class MediaPipeGestureProcessor(BaseProcessor):
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
        self.enable_sift = enable_sift
        self.enable_hands = enable_hands
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
        self.image_map_color = cv.imread("./models/colorMap.png", cv.IMREAD_COLOR)
        
        # Homography matrix
        self.H = None
        self.requires_homography = True
    
    def initialize_sift_detector(self):
        """Initialize SIFT detector for model recognition"""
        # Load the template image
        self.img_object = cv.imread(
            "./models/template.png", cv.IMREAD_GRAYSCALE
        )

        # Detect SIFT keypoints
        self.sift_detector = cv.SIFT_create()
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
        self.img_object = cv.imread("./models/template.png", cv.IMREAD_GRAYSCALE)
        
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
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Process SIFT detection if enabled
        map_detected = False
        if self.enable_sift:
            map_detected, self.H, _ = self.detect_model(frame_gray)
            if map_detected:
                output = self.draw_rect_in_image(output, self.image_map_color.shape, self.H)
        else:
            map_detected, self.H, _ = self.detect_model_with_HLoc(frame_gray)
            if map_detected:
                output = self.draw_rect_in_image(output, self.image_map_color.shape, self.H)
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

        print(f"SuperGlue: {len(matched_template_kp)} matches")

        # Need at least 4 good matches to compute homography
        if len(matched_template_kp) < 4:
            return False, None, None

        # Compute homography
        H, mask = cv.findHomography(
            matched_frame_kp, matched_template_kp,
            cv.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )

        # Count inliers
        total_inliers = np.sum(mask) if mask is not None else 0
        print(f"SuperGlue inliers: {total_inliers}")

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
            
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
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
        H, mask_out = cv.findHomography(
            scene, obj, cv.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
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
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        handedness = list()
        results = self.hands.process(image_rgb)
        coors = np.zeros((4, 3), dtype=float)
        
        # Draw the hand annotations on the image
        image_rgb.flags.writeable = True
        output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
        index_pos = None
        movement_status = None
        
        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left/right)
                if results.multi_handedness:
                    handedness.append(MessageToDict(results.multi_handedness[h])['classification'][0]['label'])
                
                # Calculate ratios for each finger
                finger_ratios = self._calculate_finger_ratios(hand_landmarks)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    output_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

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
                if (finger_ratios[1] > 0.7 and   # Index finger extended
                    finger_ratios[2] < 0.95 and  # Middle finger closed
                    finger_ratios[3] < 0.95 and  # Ring finger closed
                    finger_ratios[4] < 0.95):    # Little finger closed
                    
                    # Handle multiple hands or same hand pointing
                    if movement_status != "pointing" or (len(handedness) > 1 and handedness[1] == handedness[0]):
                        index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)
                        movement_status = "pointing"
                    else:
                        index_pos = np.append(index_pos,
                                             np.array([position[0] / position[2], position[1] / position[2], 0],
                                                     dtype=float))
                        movement_status = "too_many"
                elif movement_status != "pointing":
                    movement_status = "moving"
        
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
            
        if np.abs(position[2]) < self.Z_THRESHOLD:
            return zone, zone_desc
        else:
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
            map_image = cv.cvtColor(map_image, cv.COLOR_GRAY2BGR)
        
        # Create a mask for the map area in the output image
        h, w = map_image.shape[:2]
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Create coordinates for the four corners of the map in the map's coordinate system
        map_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        map_corners = map_corners.reshape(-1, 1, 2)
        
        # Transform the map corners to the camera view using inverse homography
        H_inv = np.linalg.inv(H)
        camera_corners = cv.perspectiveTransform(map_corners, H_inv)
        
        # Fill the polygon area in the mask
        camera_corners_int = np.int32(camera_corners.reshape(-1, 2))
        cv.fillPoly(mask, [camera_corners_int], 255)
        
        # Warp the map image to fit the perspective in the camera view
        h_image, w_image = image.shape[:2]
        warped_map = cv.warpPerspective(map_image, H_inv, (w_image, h_image))
        
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
        cv.polylines(output, [camera_corners_int], True, (0, 255, 0), 2)
        if(self.detect_pin_states(image, H)):
            text_repr = "\n".join([''.join(['●' if pixel else '○' for pixel in row]) for row in self.pin_states])
            print(text_repr, end="\r")
            return self.vis_image
        return output
    
    def detect_pin_states(self, image, H):
        """
        Detect whether pins are up (reflective) or down at hotspot locations
        
        Args:
            image (numpy.ndarray): Input image (camera frame)
            H (numpy.ndarray): Homography matrix
            
        Returns:
            tuple: (processed_image, pin_states)
                - processed_image: Image with pin states visualized
                - pin_states: Dictionary mapping hotspot IDs to states (True for up, False for down)
        """
        # Create a copy of the input image for visualization
        self.vis_image = image.copy()
        
        # Convert input image to grayscale for brightness analysis
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Get hotspots from model
        hotspots = self.model.get("hotspots", [])
        if not hotspots:
            return False
        
        # Constants for analysis
        PIN_RADIUS = 7  # Radius around pin center to analyze
        BRIGHTNESS_THRESHOLD = 150  # Threshold for considering a pin as "up" (reflective)
        CONTRAST_THRESHOLD = 30  # Threshold for contrast between pin and background
        flag = False
        # For each hotspot
        for idx, hotspot in enumerate(hotspots):
            # Get hotspot position
            if "position" not in hotspot:
                continue
            
            pin_x, pin_y = hotspot["position"]
            count = int(hotspot["hotspotTitle"])
            row, col = int((count-1)/43), (count-1)%43
            
            # Transform hotspot position to camera view
            pin_pos_map = np.array([[[pin_x, pin_y]]], dtype=np.float32)
            pin_pos_camera = cv.perspectiveTransform(pin_pos_map, np.linalg.inv(H))
            
            # Extract x,y coordinates in camera view
            cx, cy = int(pin_pos_camera[0][0][0]), int(pin_pos_camera[0][0][1])
            
            # Check if the pin position is within the image bounds
            if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
                # Extract a small region around the pin
                x_min = max(0, cx - PIN_RADIUS)
                x_max = min(image.shape[1], cx + PIN_RADIUS)
                y_min = max(0, cy - PIN_RADIUS)
                y_max = min(image.shape[0], cy + PIN_RADIUS)
                
                # Get the region of interest
                roi = gray_image[y_min:y_max, x_min:x_max]
                
                if roi.size > 0:
                    # Calculate statistics for the region
                    pin_brightness = np.mean(roi)
                    
                    # Get a slightly larger region to calculate background
                    bg_radius = PIN_RADIUS * 2
                    bg_x_min = max(0, cx - bg_radius)
                    bg_x_max = min(image.shape[1], cx + bg_radius)
                    bg_y_min = max(0, cy - bg_radius)
                    bg_y_max = min(image.shape[0], cy + bg_radius)
                    
                    # Create a mask to exclude the pin area from background calculation
                    bg_mask = np.ones((bg_y_max - bg_y_min, bg_x_max - bg_x_min), dtype=np.uint8)
                    pin_mask_y_start = y_min - bg_y_min
                    pin_mask_y_end = y_max - bg_y_min
                    pin_mask_x_start = x_min - bg_x_min
                    pin_mask_x_end = x_max - bg_x_min
                    
                    # Ensure mask indices are within bounds
                    if (0 <= pin_mask_y_start < bg_mask.shape[0] and 
                        0 <= pin_mask_y_end < bg_mask.shape[0] and
                        0 <= pin_mask_x_start < bg_mask.shape[1] and
                        0 <= pin_mask_x_end < bg_mask.shape[1]):
                        bg_mask[pin_mask_y_start:pin_mask_y_end, 
                                pin_mask_x_start:pin_mask_x_end] = 0
                        
                        # Get background region (excluding pin)
                        bg_roi = gray_image[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
                        bg_brightness = np.mean(bg_roi[bg_mask == 1]) if np.any(bg_mask) else 0
                        
                        # Calculate contrast between pin and background
                        contrast = pin_brightness - bg_brightness
                        
                        # Determine if pin is up (reflective) based on brightness and contrast
                        pin_up = (pin_brightness > BRIGHTNESS_THRESHOLD and contrast > CONTRAST_THRESHOLD)
                        if pin_up!=self.pin_states[row][col]:
                            self.pin_states[row][col] = pin_up
                            flag=True
                        
                        # Visualize the pin state
                        if pin_up:
                            # Pin is up (reflective) - draw green circle
                            cv.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 255, 0), 2)
                        else:
                            # Pin is down - draw red circle
                            cv.circle(self.vis_image, (cx, cy), PIN_RADIUS, (0, 0, 255), 2)
                        
                        # Optionally: Draw the hotspot ID for debugging
                        '''label = f"{hotspot_id}"
                        cv.putText(vis_image, label, (cx + PIN_RADIUS, cy), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)'''
        
        return flag
    
    def reset_homography(self):
        """Reset homography to force recalculation"""
        self.requires_homography = True
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if self.enable_hands:
            self.hands.close()
    
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