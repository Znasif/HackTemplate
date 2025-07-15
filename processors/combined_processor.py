import cv2
import numpy as np
import mediapipe as mp
import torch
import asyncio
import time
import json
import colorsys
import os
import glob
from collections import deque
from typing import Dict, Optional, Tuple, List, Any, Union
from scipy import stats
from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.superglue import SuperGlue
from google.protobuf.json_format import MessageToDict
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from .base_processor import BaseProcessor
from .extract import ControlPanelRegionGenerator


class InMemoryTemplateCache:
    """Stores all template data as numpy arrays in memory"""
    def __init__(self, max_templates=20):
        self.templates = {}
        self.max_templates = max_templates
        self.access_order = deque(maxlen=max_templates)  # LRU tracking
        
    def add_template(self, template_id: str, panel_image: np.ndarray, 
                     color_map: np.ndarray, regions_array: np.ndarray, 
                     homography: Optional[np.ndarray] = None):
        """Store template entirely in memory using numpy arrays"""
        
        # Evict oldest if at capacity
        if len(self.templates) >= self.max_templates and template_id not in self.templates:
            oldest = self.access_order[0]
            del self.templates[oldest]
            
        self.templates[template_id] = {
            'panel_image': panel_image.copy(),
            'color_map': color_map.copy(),
            'regions_array': regions_array.copy(),  # Structured numpy array
            'homography': homography.copy() if homography is not None else None,
            'timestamp': time.time()
        }
        
        # Update access order
        if template_id in self.access_order:
            self.access_order.remove(template_id)
        self.access_order.append(template_id)
        
    def get_template(self, template_id: str):
        """Get template and update access order"""
        if template_id in self.templates:
            # Update access order
            self.access_order.remove(template_id)
            self.access_order.append(template_id)
            return self.templates[template_id]
        return None
    
    def get_all_templates(self):
        """Get all templates for matching"""
        return self.templates


class MultiTemplateTracker:
    """Tracks multiple templates simultaneously when multiple panels are visible"""
    def __init__(self, template_cache: InMemoryTemplateCache):
        self.template_cache = template_cache
        self.active_templates = {}  # template_id -> tracking_data
        self.tracking_threshold = 0.5  # Reasonable threshold for tracking
        self.max_tracking_distance = 50.0  # Maximum distance for tracking continuity
        
    def get_all_templates(self) -> List[str]:
        """Get all available template IDs"""
        return list(self.template_cache.get_all_templates().keys())
    
    def update_tracking(self, frame_matches: List[Dict]) -> Dict[str, Dict]:
        """Update tracking for multiple templates based on frame matches"""
        current_templates = {}
        
        for match in frame_matches:
            template_id = match['template_id']
            confidence = match['confidence']
            homography = match['homography']
            
            # Check if this is a good enough match
            if confidence >= self.tracking_threshold:
                current_templates[template_id] = {
                    'homography': homography,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
        
        # Update active templates
        self.active_templates = current_templates
        
        return current_templates
    
    def get_active_templates(self) -> Dict[str, Dict]:
        """Get currently active templates"""
        return self.active_templates
    
    def get_template_data(self, template_id: str) -> Optional[Dict]:
        """Get template data for a specific template ID"""
        return self.template_cache.get_template(template_id)


class CombinedRegionProcessor(BaseProcessor):
    """Combined processor with template selection and region processing"""
    def __init__(self, enable_sift=True, enable_hands=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super().__init__()
        self.locked_template_id = None          # Currently locked template ID
        self.locked_template_confidence = 0.0   # Confidence of locked template
        self.frames_without_hand = 0           # Frames since last hand detection
        self.max_frames_without_hand = 10      # Release lock after this many frames
        self.min_lock_confidence = 0.3         # Minimum confidence to maintain lock

        
        # Add template tracking endpoints
        @self.app.get("/templates")
        async def get_templates():
            """Get information about cached templates"""
            return self.get_template_info()
        
        @self.app.get("/active_templates")
        async def get_active_templates():
            """Get currently active templates being tracked"""
            return {"active_templates": self.template_tracker.get_active_templates()}
        
        @self.app.get("/debug")
        async def debug_info():
            """Get debug information about the processor state"""
            return {
                "total_templates": len(self.template_cache.get_all_templates()),
                "active_templates": list(self.active_templates.keys()),
                "template_cache_keys": list(self.template_cache.get_all_templates().keys()),
                "tracking_threshold": self.template_tracker.tracking_threshold
            }
        
        @self.app.get("/test_proximity")
        async def test_proximity(x: int = 200, y: int = 300):
            """Test proximity detection with a specific hand position"""
            hand_pos = np.array([x, y])
            closest = self.find_closest_template(hand_pos)
            if closest:
                template_id, template_data, distance = closest
                return {
                    "hand_position": [x, y],
                    "closest_template": template_id,
                    "distance": distance,
                    "active_templates": list(self.active_templates.keys())
                }
            else:
                return {
                    "hand_position": [x, y],
                    "closest_template": None,
                    "active_templates": list(self.active_templates.keys())
                }
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Template management (from combined_processor)
        self.template_cache = InMemoryTemplateCache()
        self.template_tracker = MultiTemplateTracker(self.template_cache)
        self.active_templates = {}  # Currently tracked templates
        self.template_regions = {}  # Regions for each template
        self.template_color_maps = {}  # Color maps for each template
        
        # State management
        self.is_tracking = False
        self.H = None  # Current homography
        self.last_frame_time = 0
        self.frames_without_detection = 0
        
        # Tracking stability
        self.tracking_frames = 0  # Count consecutive tracking frames
        self.lost_frames = 0      # Count consecutive lost frames
        self.min_tracking_frames = 2  # Minimum frames to establish tracking
        self.max_lost_frames = 5      # Maximum frames before considering tracking lost
        
        # Region processing settings (from region1_processor)
        self.enable_sift = enable_sift
        self.enable_hands = enable_hands
        self.current_desc = ""
        
        # Initialize detection components
        self._initialize_detectors()
        
        # Load pre-existing templates at initialization
        self._load_existing_templates()
        
        # Initialize zone interaction (from region1_processor)
        self.initialize_interaction_policy()
        
        # Initialize movement filters (from region1_processor)
        self.movement_filter = self.MovementMedianFilter()
        self.gesture_detector = self.GestureDetector()
    
    def _load_existing_templates(self):
        """Load pre-existing panel templates from extract.py outputs"""
        print("🔄 Loading existing panel templates...")
        
        # Find all panel output directories in models/cars/
        cars_dir = "./models/cars"
        panel_output_pattern = os.path.join(cars_dir, "panel*_output")
        panel_outputs = glob.glob(panel_output_pattern)
        
        if not panel_outputs:
            print("⚠️ No panel output directories found in models/cars/")
            # Try to process panels if no outputs exist
            self._process_panels_if_needed()
            return
        
        print(f"📁 Found {len(panel_outputs)} panel output directories")
        
        # Load each panel's data
        successful_templates = 0
        for output_dir in panel_outputs:
            try:
                # Extract panel name from directory
                panel_name = os.path.basename(output_dir).replace('_output', '')
                print(f"🔄 Loading {panel_name}...")
                
                # Load the template data
                result = self._load_panel_from_files(output_dir, panel_name)
                
                if result:
                    # Add to template cache
                    self.template_cache.add_template(
                        panel_name,
                        result['panel_image'],
                        result['color_map'],
                        result['regions_array'],
                        result.get('homography')
                    )
                    
                    successful_templates += 1
                    print(f"✅ {panel_name} loaded with {len(result['regions_array'])} regions")
                    print(f"   Panel image shape: {result['panel_image'].shape}")
                    print(f"   Color map shape: {result['color_map'].shape}")
                    # Log region details
                    for i, region in enumerate(result['regions_array']):
                        if region['text'] and len(region['text']) > 0:
                            print(f"   Region {i+1}: '{region['text']}' at {region['bbox']}")
                else:
                    print(f"❌ Failed to load {panel_name}")
                    
            except Exception as e:
                print(f"❌ Error loading {output_dir}: {e}")
                continue
        
        if successful_templates > 0:
            print(f"🎯 Template loading complete. Successfully loaded {successful_templates}/{len(panel_outputs)} templates")
        else:
            print("❌ No templates were successfully loaded!")
    
    def _process_panels_if_needed(self):
        """Process panels using extract.py if no outputs exist"""
        print("🔄 No panel outputs found, processing panels with extract.py...")
        
        # Find panel images
        cars_dir = "./models/cars"
        panel_pattern = os.path.join(cars_dir, "panel*.jpg")
        panel_images = glob.glob(panel_pattern)
        
        if not panel_images:
            print("⚠️ No panel images found to process")
            return
        
        print(f"📁 Found {len(panel_images)} panel images to process")
        
        # Initialize the extractor
        extractor = ControlPanelRegionGenerator(
            florence_model_name="microsoft/Florence-2-large",
            sam_model_path="./models/sam2.1_l.pt"
        )
        
        # Process each panel
        for panel_path in panel_images:
            try:
                panel_name = os.path.basename(panel_path).replace('.jpg', '')
                output_dir = os.path.join(cars_dir, f"{panel_name}_output")
                
                print(f"🔄 Processing {panel_name}...")
                extractor.process_control_panel(panel_path, output_dir)
                print(f"✅ {panel_name} processed")
                
            except Exception as e:
                print(f"❌ Error processing {panel_path}: {e}")
                continue
        
        # Now try to load the newly created outputs
        self._load_existing_templates()
    
    def _load_panel_from_files(self, output_dir: str, panel_name: str) -> Optional[Dict]:
        """Load panel data from extract.py output files"""
        try:
            # Load regions.json
            regions_path = os.path.join(output_dir, "regions.json")
            if not os.path.exists(regions_path):
                print(f"   ❌ regions.json not found in {output_dir}")
                return None
            
            with open(regions_path, 'r') as f:
                regions_data = json.load(f)
            
            # Load ISOLATED panel image - this is what the regions are based on
            isolated_panel_path = os.path.join(output_dir, f"{panel_name}_isolated_panel.jpg")
            if not os.path.exists(isolated_panel_path):
                print(f"   ❌ isolated panel image not found: {isolated_panel_path}")
                return None
            
            panel_image = cv2.imread(isolated_panel_path)
            if panel_image is None:
                print(f"   ❌ Failed to load isolated panel image: {isolated_panel_path}")
                return None
            
            # Load color map
            color_map_path = os.path.join(output_dir, "colorMap.png")
            if not os.path.exists(color_map_path):
                print(f"   ❌ colorMap.png not found in {output_dir}")
                return None
            
            color_map = cv2.imread(color_map_path)
            if color_map is None:
                print(f"   ❌ Failed to load color map: {color_map_path}")
                return None
            
            # Convert regions data to numpy array format
            regions_array = self._convert_regions_to_array(regions_data['regions'])
            
            return {
                'panel_image': panel_image,
                'color_map': color_map,
                'regions_array': regions_array,
                'homography': None  # No homography stored in files
            }
            
        except Exception as e:
            print(f"   ❌ Error loading panel from files: {e}")
            return None
    
    def _convert_regions_to_array(self, regions_data: List[Dict]) -> np.ndarray:
        """Convert regions data to structured numpy array"""
        # Define the dtype for the structured array
        dtype = [
            ('detection_id', 'U50'),
            ('text', 'U200'),
            ('bbox', '4i4'),  # x1, y1, x2, y2
            ('color', '3i4'),  # BGR
            ('confidence', 'f4')
        ]
        
        regions_array = np.empty(len(regions_data), dtype=dtype)
        
        for i, region in enumerate(regions_data):
            regions_array[i] = (
                region.get('detection_id', f'region_{i}'),
                region.get('text', ''),
                tuple(region.get('bbox', [0, 0, 0, 0])),
                tuple(region.get('color', [0, 0, 0])),
                region.get('confidence', 0.0)
            )
        
        return regions_array
    
    def _initialize_detectors(self):
        """Initialize detection components"""
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection
        if self.enable_hands:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Initialize SIFT model detector
        if self.enable_sift:
            self.initialize_sift_detector()
        else:
            self.initialize_HLoc_detector()
    
    def initialize_sift_detector(self):
        """Initialize SIFT detector for model recognition"""
        # We'll load template dynamically based on selection
        self.sift_detector = cv2.SIFT_create()
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
        self.superpoint = self.superpoint.to(self.device)
        self.superglue = self.superglue.to(self.device)
        
        # Minimum number of inliers for a valid detection
        self.MIN_INLIER_COUNT = 15
    
    def initialize_interaction_policy(self):
        """Initialize zone interaction policy"""
        self.ZONE_FILTER_SIZE = 10
        self.Z_THRESHOLD = 2.0
        self.zone_filter = -1 * np.ones(self.ZONE_FILTER_SIZE, dtype=int)
        self.zone_filter_cnt = 0
    

    # Minimal fix - just replace these two methods in your CombinedRegionProcessor class:

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Process frame with robust, proximity-based template locking for hand interaction.
        """
        output = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        region_name = "No interaction"

        # 1. Match all visible templates and update tracking
        frame_matches = []
        all_templates = self.template_tracker.get_all_templates()
        for template_id in all_templates:
            template_data = self.template_cache.get_template(template_id)
            if template_data:
                if self.enable_sift:
                    detected, H, conf = self.detect_model_with_template(frame_gray, template_data)
                else:
                    detected, H, conf = self.detect_model_with_HLoc(frame_gray, template_data)
                
                if detected and H is not None:
                    frame_matches.append({
                        'template_id': template_id, 'homography': H,
                        'confidence': conf, 'template_data': template_data
                    })
        
        self.active_templates = self.template_tracker.update_tracking(frame_matches)

        # 2. Render overlays for all visible templates
        for match in frame_matches:
            output = self.render_regions_overlay(output, match['template_data'], match['homography'], match['template_id'])

        # 3. --- HAND INTERACTION LOGIC ---
        if self.enable_hands and self.active_templates:
            # 3a. Detect hand ONCE to get its camera-space coordinates
            hand_camera_pos = self._get_hand_camera_position(frame)
            
            # 3b. Determine which template to interact with
            template_for_hand = None
            if self.locked_template_id:
                # If a template is locked, check if it's still active
                if self.locked_template_id in self.active_templates:
                    template_for_hand = {
                        'template_id': self.locked_template_id,
                        **self.active_templates[self.locked_template_id],
                        'template_data': self.template_cache.get_template(self.locked_template_id)
                    }
                else:
                    # Locked template is no longer visible, release the lock
                    self.locked_template_id = None
            
            # 3c. If no lock, find the closest template to the hand to acquire a new lock
            if not self.locked_template_id and hand_camera_pos is not None:
                template_for_hand = self._find_closest_template_to_hand(hand_camera_pos, frame_matches)
                if template_for_hand:
                    self.locked_template_id = template_for_hand['template_id']
                    print(f"✅ Acquired lock on template: {self.locked_template_id}")

            # 3d. Process gesture IF a template has been selected (either locked or newly found)
            if template_for_hand:
                gesture_loc, gesture_status, output = self.detect_hands(
                    frame, output, template_for_hand['homography']
                )

                if gesture_loc is not None and gesture_status == "pointing":
                    self.frames_without_hand = 0
                    region_name = self.get_region_name(gesture_loc, template_for_hand['template_id'], template_for_hand['template_data'])
                else:
                    self.frames_without_hand += 1
            else:
                 # This case handles when a hand is not visible
                 self.frames_without_hand += 1
            
            # 3e. Release lock if no hand is detected for too long
            if self.locked_template_id and self.frames_without_hand > self.max_frames_without_hand:
                print(f"⭕ Releasing lock on {self.locked_template_id} (timeout)")
                self.locked_template_id = None
                
        # Add lock status to output for debugging
        if self.locked_template_id:
            cv2.putText(output, f"Locked: {self.locked_template_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return output, region_name
    
    def _get_hand_camera_position(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects a hand and returns the index finger's position in camera pixel coordinates."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Use the first hand found
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            return np.array([cx, cy])
            
        return None

    def _find_closest_template_to_hand(self, hand_camera_pos: np.ndarray, frame_matches: List[Dict]) -> Optional[Dict]:
        """
        Finds which template the hand is closest to. This is more robust than a strict in-bounds check.
        """
        closest_template = None
        min_dist = float('inf')

        for match in frame_matches:
            H = match['homography']
            template_data = match['template_data']
            h, w = template_data['panel_image'].shape[:2]

            # Transform hand position from camera space to this template's space
            cam_pos_homogeneous = np.array([hand_camera_pos[0], hand_camera_pos[1], 1.0])
            template_pos_homogeneous = np.matmul(H, cam_pos_homogeneous)
            
            # De-homogenize to get (x, y) in the template's coordinate system
            tx = template_pos_homogeneous[0] / template_pos_homogeneous[2]
            ty = template_pos_homogeneous[1] / template_pos_homogeneous[2]
            
            # Calculate the distance from the point to the center of the template rectangle
            # This is a simple and effective proximity metric.
            center_x, center_y = w / 2, h / 2
            dist = np.sqrt((tx - center_x)**2 + (ty - center_y)**2)

            # We only consider a template if the hand is reasonably close to it.
            # Here, we check if the distance to the center is less than the template's diagonal
            # (a generous boundary) and if it's the closest so far.
            max_dist_threshold = np.sqrt(w**2 + h**2) 
            if dist < min_dist and dist < max_dist_threshold:
                min_dist = dist
                closest_template = match
        
        return closest_template
    
    def _find_template_for_hand(self, frame: np.ndarray, frame_matches: List[Dict]) -> Optional[Dict]:
        """
        Find which template the hand is currently over by checking each template
        Returns the first template where hand is detected within bounds
        """
        for match in frame_matches:
            template_id = match['template_id']
            H = match['homography']
            template_data = match['template_data']
            
            # Do a quick hand detection with this template's homography
            # Using a simplified version that just checks for hand presence
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger position in camera coordinates
                    index_x = hand_landmarks.landmark[8].x * frame.shape[1]
                    index_y = hand_landmarks.landmark[8].y * frame.shape[0]
                    
                    # Transform to this template's coordinates
                    position = np.matmul(H, np.array([index_x, index_y, 1]))
                    template_x = position[0] / position[2]
                    template_y = position[1] / position[2]
                    
                    # Check if within this template's bounds
                    color_map = template_data['color_map']
                    h, w = color_map.shape[:2]
                    
                    if 0 <= template_x < w and 0 <= template_y < h:
                        print(f"DEBUG: Hand found in template {template_id} bounds")
                        return match
        
        print("DEBUG: Hand not found in any template bounds")
        return None
    
    def detect_model_with_template(self, frame_gray: np.ndarray, template_data: Dict) -> Tuple[bool, Optional[np.ndarray], float]:
        """Detect model in frame using SIFT with specific template"""
        panel_image = template_data['panel_image']
        panel_gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT keypoints for template
        keypoints_obj, descriptors_obj = self.sift_detector.detectAndCompute(panel_gray, mask=None)
        
        if descriptors_obj is None:
            return False, None, 0.0
        
        # Detect SIFT keypoints for frame
        keypoints_scene, descriptors_scene = self.sift_detector.detectAndCompute(frame_gray, mask=None)
        
        if descriptors_scene is None:
            return False, None, 0.0
        
        # Match keypoints
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors_obj, descriptors_scene, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.MIN_INLIER_COUNT:
            return False, None, 0.0
        
        # Calculate confidence based on number of good matches
        confidence = min(len(good_matches) / 100.0, 1.0)  # Normalize to 0-1
        
        # Extract matched keypoints
        src_pts = np.float32([keypoints_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return False, None, 0.0
        
        return True, H, confidence
    
    def detect_model_with_HLoc(self, frame_gray: np.ndarray, template_data: Dict) -> Tuple[bool, Optional[np.ndarray], float]:
        """Detect model in frame using SuperPoint features and SuperGlue matcher"""
        panel_image = template_data['panel_image']
        panel_gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
        
        # Process template image for SuperPoint
        template_tensor = torch.from_numpy(panel_gray).float() / 255.
        template_tensor = template_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Extract template features
        with torch.no_grad():
            template_data_features = self.superpoint({'image': template_tensor})
        
        # Preprocess frame for SuperPoint
        frame_tensor = torch.from_numpy(frame_gray).float() / 255.
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Extract features from current frame
        with torch.no_grad():
            frame_data = self.superpoint({'image': frame_tensor})

            # Create SuperGlue input dict with required keys
            superglue_input = {
                'keypoints0': template_data_features['keypoints'][0].unsqueeze(0),
                'scores0': template_data_features['scores'][0].unsqueeze(0),
                'descriptors0': template_data_features['descriptors'][0].unsqueeze(0),
                'keypoints1': frame_data['keypoints'][0].unsqueeze(0),
                'scores1': frame_data['scores'][0].unsqueeze(0),
                'descriptors1': frame_data['descriptors'][0].unsqueeze(0),
                'image0': template_tensor,
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
        template_kp = template_data_features['keypoints'][0].cpu().numpy()
        frame_kp = frame_data['keypoints'][0].cpu().numpy()

        # Get matched pairs
        matched_template_kp = template_kp[valid]
        matched_frame_kp = frame_kp[matches[valid]]

        # Need at least 4 good matches to compute homography
        if len(matched_template_kp) < 4:
            return False, None, 0.0

        # Calculate confidence based on number of matches and their scores
        confidence = min(len(matched_template_kp) / 50.0, 1.0)  # Normalize to 0-1

        # Compute homography
        H, mask = cv2.findHomography(
            matched_frame_kp, matched_template_kp,
            cv2.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )
        
        if H is None:
            return False, None, 0.0
        
        return True, H, confidence
    
    def detect_hands(self, image, output_image, H):
        """Detect hands and process gestures (same as region1_processor)"""
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
                
                # Get index finger position (landmark 8)
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
        Calculate ratios for each finger to determine if they're extended (same as region1_processor)
        
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
        Calculate ratio to determine if finger is extended (same as region1_processor)
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
    

    
    def push_gesture(self, position, template_id=None, template_data=None):
        """Process gesture position and determine region interaction for specific template"""
        # Apply movement filter
        filtered_position = self.movement_filter.push_position(position)
        
        if filtered_position is None:
            return "No stable position detected"
        
        # Get zone information for specific template
        zone_info = self.get_zone(filtered_position, template_id, template_data)
        
        # Apply gesture detection
        gesture_info = self.gesture_detector.push_position(filtered_position)
        
        return f"Template {template_id}: Zone {zone_info}, Gesture {gesture_info}"
    
    def get_zone(self, position, template_id=None, template_data=None):
        """Get zone information for a position in specific template"""
        # Get template data and homography
        if template_data is None and template_id is not None:
            template_data = self.template_cache.get_template(template_id)
        
        if template_data is None:
            return "No template data available"
        
        # Get homography for this template
        if template_id in self.active_templates:
            H = self.active_templates[template_id]['homography']
        else:
            return "Template not currently tracked"
        
        if H is None:
            return "No homography available"
        
        # Transform position to template coordinates
        point = np.array([[position]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, H)
        
        if transformed_point is None or len(transformed_point) == 0:
            return "Transform failed"
        
        x, y = transformed_point[0][0]
        
        # Check bounds
        color_map = template_data['color_map']
        h, w = color_map.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return "Position out of bounds"
        
        # Get color at position
        color = color_map[int(y), int(x)]
        color_tuple = tuple(color)
        
        # Get region from color
        region_info = self.get_region_from_color(color_tuple, template_data)
        
        return region_info
    
    def find_closest_template(self, hand_position: np.ndarray) -> Optional[Tuple[str, Dict, float]]:
        """Find the template closest to the hand position (hand_position is in template coordinates)"""
        if not self.active_templates:
            print(f"DEBUG: No active templates found")
            return None
        
        print(f"DEBUG: Hand position (template coords): {hand_position}")
        print(f"DEBUG: Active templates: {list(self.active_templates.keys())}")
        
        closest_template = None
        min_distance = float('inf')
        max_distance_threshold = 100.0  # Distance threshold in template coordinates
        
        for template_id, tracking_data in self.active_templates.items():
            print(f"DEBUG: Checking template {template_id}")
            
            template_data = self.template_cache.get_template(template_id)
            if template_data is None:
                print(f"DEBUG: No template data for {template_id}")
                continue
            
            # Check if hand position is within template bounds
            color_map = template_data['color_map']
            h, w = color_map.shape[:2]
            
            x, y = hand_position[0], hand_position[1]
            print(f"DEBUG: Hand at ({x:.2f}, {y:.2f}) in template {template_id} bounds {w}x{h}")
            
            if 0 <= x < w and 0 <= y < h:
                # Hand is within template bounds, calculate distance to center
                center_x, center_y = w / 2, h / 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                print(f"DEBUG: Distance to {template_id} center: {distance:.2f} (threshold: {max_distance_threshold})")
                
                if distance < min_distance and distance < max_distance_threshold:
                    min_distance = distance
                    closest_template = (template_id, template_data, distance)
                    print(f"DEBUG: New closest template: {template_id} at distance {distance:.2f}")
                else:
                    print(f"DEBUG: Template {template_id} rejected - distance {distance:.2f} >= threshold {max_distance_threshold}")
            else:
                print(f"DEBUG: Hand position ({x:.2f}, {y:.2f}) outside template {template_id} bounds")
        
        if closest_template:
            print(f"DEBUG: Selected template: {closest_template[0]} at distance {closest_template[2]:.2f}")
        else:
            print(f"DEBUG: No suitable template found")
        
        return closest_template
    
    def get_region_name(self, hand_position: np.ndarray, template_id: str, template_data: Dict) -> str:
        """Get region name for hand position in specific template (hand_position is in template coordinates)"""
        # Hand position is already in template coordinates, no transformation needed
        x, y = hand_position[0], hand_position[1]
        
        # Check bounds
        color_map = template_data['color_map']
        h, w = color_map.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return "Position out of bounds"
        
        # Get color at position
        color = color_map[int(y), int(x)]
        color_tuple = tuple(color)
        
        print(f"DEBUG: Color at position ({x:.2f}, {y:.2f}): {color_tuple}")
        
        # Get region from color
        region_info = self.get_region_from_color(color_tuple, template_data)
        
        return region_info
    
    def get_region_from_color(self, color, template_data=None):
        """Get region information from color for specific template"""
        if template_data is None:
            return "No template data available"
        
        regions_array = template_data['regions_array']
        if regions_array is None:
            return "No regions available"
        
        # Find region with matching color
        for region in regions_array:
            region_color = tuple(region['color'])
            if region_color == color:
                # Use 'class' field if available, otherwise use 'text' or 'detection_id'
                if 'class' in region and region['class']:
                    return region['class']
                elif 'text' in region and region['text']:
                    return region['text']
                else:
                    return f"Region {region['detection_id']}"
        
        return "Unknown region"
    
    def render_regions_overlay(self, frame, template_data, H, template_id=None):
        """
        Renders regions overlay using homography projection (same as region1_processor)
        
        Args:
            frame (numpy.ndarray): Camera frame to draw on
            template_data (dict): Template data containing color map
            H (numpy.ndarray): Homography matrix
            template_id (str): Optional template ID for labeling
            
        Returns:
            numpy.ndarray: Frame with regions overlay
        """
        # Create a copy of the input frame
        output = frame.copy()
        
        # Check if homography matrix exists
        if H is None:
            return output
        
        # Create a copy of the color map
        map_image = template_data['color_map'].copy()
        
        # Create a mask for the map area in the output image
        h, w = map_image.shape[:2]
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Create coordinates for the four corners of the map
        map_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        map_corners = map_corners.reshape(-1, 1, 2)
        
        # Transform the map corners to the camera view using inverse homography
        H_inv = np.linalg.inv(H)
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
        
        # Draw the outline to show the boundaries clearly
        outline_color = (0, 255, 0)  # Green outline
        if template_id:
            # Generate consistent color based on template ID for outline
            color_hash = hash(template_id) % 360
            hue = color_hash / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            outline_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR
        
        cv2.polylines(output, [camera_corners_int], True, outline_color, 2)
        
        # Add template ID label if provided
        if template_id:
            # Position label at top-left corner
            label_pos = camera_corners[0][0]
            cv2.putText(output, template_id, (int(label_pos[0]), int(label_pos[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 2)
        
        return output
    
    def draw_rect_in_image(self, image, sz, H, template_id=None):
        """Draw template overlay on image with optional template ID label"""
        if H is None:
            return image
        
        h, w = sz[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners = corners.reshape(-1, 1, 2)
        
        # Transform corners
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Draw rectangle with different colors for different templates
        if template_id:
            # Generate consistent color based on template ID
            color_hash = hash(template_id) % 360
            hue = color_hash / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR
        else:
            color = (0, 255, 0)  # Default green
        
        cv2.polylines(image, [np.int32(transformed_corners)], True, color, 2)
        
        # Add template ID label if provided
        if template_id:
            # Position label at top-left corner
            label_pos = transformed_corners[0][0]
            cv2.putText(image, template_id, (int(label_pos[0]), int(label_pos[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def get_template_info(self) -> Dict:
        """Get information about cached templates"""
        templates = self.template_cache.get_all_templates()
        active_templates = self.template_tracker.get_active_templates()
        
        info = {
            "total_templates": len(templates),
            "active_templates": list(active_templates.keys()),
            "tracking_info": active_templates,
            "templates": {}
        }
        
        for template_id, template_data in templates.items():
            info["templates"][template_id] = {
                "regions_count": len(template_data['regions_array']),
                "panel_shape": template_data['panel_image'].shape,
                "color_map_shape": template_data['color_map'].shape,
                "timestamp": template_data['timestamp'],
                "is_active": template_id in active_templates
            }
        
        return info
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        return point_cloud_data, {"message": "CombinedRegionProcessor received point cloud data but does not process it further."}
    
    class MovementMedianFilter:
        """Filter for stabilizing hand movement"""
        def __init__(self, window_size=5):
            self.window_size = window_size
            self.positions = deque(maxlen=window_size)
        
        def push_position(self, position):
            """Add position and return filtered result"""
            self.positions.append(position)
            
            if len(self.positions) < self.window_size:
                return None
            
            # Calculate median position
            positions_array = np.array(list(self.positions))
            median_position = np.median(positions_array, axis=0)
            
            return median_position
    
    class GestureDetector:
        """Detect gestures from hand positions"""
        def __init__(self):
            self.last_position = None
            self.movement_threshold = 10.0
        
        def push_position(self, position):
            """Process position and detect gestures"""
            if self.last_position is None:
                self.last_position = position
                return "initial_position"
            
            # Calculate movement
            movement = np.linalg.norm(position - self.last_position)
            
            if movement > self.movement_threshold:
                gesture = "moving"
            else:
                gesture = "stationary"
            
            self.last_position = position
            return gesture

# Create processor instance
processor = CombinedRegionProcessor()
app = processor.app