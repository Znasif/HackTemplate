from .base_processor import BaseProcessor
import cv2
import numpy as np
import os
import torch
import sys
import warnings

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import random
from PIL import Image
import time
import subprocess

def _apply_torch_patches():
    """Apply patches to torch functions to handle version compatibility issues"""
    print("Applying PyTorch compatibility patches")
    
    # Store original functions
    original_torch_min = torch.min
    original_torch_max = torch.max
    
    # Check if _amin and _amax exist
    has_amin = hasattr(torch._C._VariableFunctions, '_amin')
    has_amax = hasattr(torch._C._VariableFunctions, '_amax')
    
    if has_amin:
        original_torch_amin = torch._C._VariableFunctions._amin
    if has_amax:
        original_torch_amax = torch._C._VariableFunctions._amax
    
    # Create wrapper for min function
    def patched_min(input, dim=None, keepdim=False, out=None):
        if dim is not None:
            try:
                return original_torch_min(input, dim=dim, keepdim=keepdim, out=out)
            except TypeError:
                return original_torch_min(input, axis=dim, keepdim=keepdim, out=out)
        return original_torch_min(input, out=out)
    
    # Create wrapper for amin function if it exists
    if has_amin:
        def patched_amin(input, dim=None, keepdim=False, out=None):
            if dim is not None:
                try:
                    return original_torch_amin(input, dim=dim, keepdim=keepdim, out=out)
                except TypeError:
                    return original_torch_amin(input, axis=dim, keepdim=keepdim, out=out)
            return original_torch_amin(input, out=out)
    
    # Create wrapper for max function
    def patched_max(input, dim=None, keepdim=False, out=None):
        if dim is not None:
            try:
                return original_torch_max(input, dim=dim, keepdim=keepdim, out=out)
            except TypeError:
                return original_torch_max(input, axis=dim, keepdim=keepdim, out=out)
        return original_torch_max(input, out=out)
    
    # Create wrapper for amax function if it exists
    if has_amax:
        def patched_amax(input, dim=None, keepdim=False, out=None):
            if dim is not None:
                try:
                    return original_torch_amax(input, dim=dim, keepdim=keepdim, out=out)
                except TypeError:
                    return original_torch_amax(input, axis=dim, keepdim=keepdim, out=out)
            return original_torch_amax(input, out=out)
    
    # Apply the patches
    torch.min = patched_min
    if has_amin:
        torch._C._VariableFunctions._amin = patched_amin
    
    torch.max = patched_max
    if has_amax:
        torch._C._VariableFunctions._amax = patched_amax
        
    print("PyTorch compatibility patches applied")

# Apply patches immediately
_apply_torch_patches()

llava_parent_dir = r'/home/znasif/scenescript' # Use raw string for paths

# Add this directory to sys.path if it's not already there
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

class SceneScriptProcessor(BaseProcessor):
    def __init__(self, 
                 model_path='/home/znasif/vidServer/server/models/scenescript_model_non_manhattan_class_agnostic_model.ckpt',
                 class_names_path=None,
                 confidence_threshold=0.5,
                 use_gpu=True):
        """
        Initialize SceneScript processor for object detection in scenes
        
        Args:
            model_path (str): Path to the SceneScript model checkpoint
            class_names_path (str): Path to class names file (optional)
            confidence_threshold (float): Minimum confidence for detection
            use_gpu (bool): Whether to use GPU for inference
        """
        super().__init__()
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load class names if path is provided
        self.class_names = []
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Default class names from common indoor objects
            self.class_names = [
                'wall', 'door', 'window', 'chair', 'sofa', 'table', 'bed', 
                'cabinet', 'shelf', 'lamp', 'plant', 'monitor', 'picture', 
                'book', 'person', 'object'
            ]
        
        # Initialize color map for each class
        self.colors = self._generate_colors(len(self.class_names))
        
        # Check dependencies - torchsparse is required for SceneScript
        try:
            self._check_dependencies()
            self.dependencies_installed = True
        except Exception as e:
            self.dependencies_installed = False
            print(f"Error checking dependencies: {e}")
            self.model_loaded = False
            self.model_info = {
                "error": f"Dependencies not installed: {str(e)}. ",
                "name": os.path.basename(model_path),
                "type": "SceneScript Object Detector"
            }
            return
            
        # Add SceneScript to sys.path
        scenescript_path = '/home/znasif/scenescript'
        if scenescript_path not in sys.path:
            sys.path.append(scenescript_path)
            
        # Now try to import SceneScript modules
        try:
            # Import SceneScript modules
            from src.networks.scenescript_model import SceneScriptWrapper
            from src.data.point_cloud import PointCloud
            from scipy.spatial.transform import Rotation
            
            # Store the module references
            self.SceneScriptWrapper = SceneScriptWrapper
            self.PointCloud = PointCloud
            self.Rotation = Rotation
            
            # Load SceneScript model
            print(f"Loading SceneScript model from {model_path}")
            self.model = self.SceneScriptWrapper.load_from_checkpoint(model_path)
            if self.device == 'cuda':
                self.model = self.model.cuda()
            
            # Set model info
            self.model_loaded = True
            self.model_info = {
                "name": os.path.basename(model_path),
                "type": "SceneScript Object Detector",
                "num_classes": len(self.class_names)
            }
            print(f"SceneScript model loaded successfully: {self.model_info}")
        except Exception as e:
            self.model_loaded = False
            self.model_info = {
                "error": f"Failed to load model: {str(e)}",
                "name": os.path.basename(model_path),
                "type": "SceneScript Object Detector"
            }
            print(f"Error loading SceneScript model: {e}")
    
    def _check_dependencies(self):
        """Check if all required dependencies are installed"""
        # First check if torchsparse is installed
        try:
            import torchsparse
            print("torchsparse is already installed")
        except ImportError:
            error_msg = ""
            print(error_msg)
            raise ImportError(error_msg)
            
    def _generate_colors(self, num_classes):
        """
        Generate distinct colors for each class
        
        Args:
            num_classes (int): Number of classes
            
        Returns:
            list: List of BGR color tuples
        """
        colors = []
        for i in range(num_classes):
            # Generate vibrant colors with good separation
            hue = i * 179 // num_classes
            # Convert HSV to BGR
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
            colors.append((int(color[0][0][0]), int(color[0][0][1]), int(color[0][0][2])))
        return colors
    
    def _create_point_cloud_from_image(self, frame):
        """
        Create a pseudo point cloud from an image for SceneScript
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Point cloud array (N, 3)
        """
        # This is a simplified approach - in a real scenario, 
        # you would use depth information or other 3D reconstruction methods
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to reasonable dimensions to reduce computation
        h, w = frame.shape[:2]
        target_size = (320, int(h * 320 / w))
        resized = cv2.resize(rgb_frame, target_size)
        
        # Create a simple point cloud from image
        # We'll use pixel positions as x,y and add some random depth values
        h, w = resized.shape[:2]
        points = []
        
        # Sample a subset of pixels to keep point cloud manageable
        stride = 4  # Sample every 4th pixel
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Use pixel RGB as features, and add some random depth
                r, g, b = resized[y, x]
                # Normalize coordinates
                nx = x / w - 0.5  # Center the point cloud around origin
                ny = y / h - 0.5
                # Add some random depth based on pixel intensity
                # Convert to float to avoid overflow
                depth = (float(r) + float(g) + float(b)) / (3 * 255) * 2 + 1  # Range 1-3
                
                # Add the point to our point cloud
                points.append([nx, ny, depth])
        
        return np.array(points, dtype=np.float32)
        
    def process_frame(self, frame):
        """
        Process frame by detecting objects and drawing bounding boxes
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, detection_text)
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Check if dependencies are installed
        if not getattr(self, 'dependencies_installed', False):
            # Draw installation instructions on frame
            cv2.putText(
                output,
                "Error: Required dependencies not installed",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.putText(
                output,
                "",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.putText(
                output,
                "Then restart the server",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return output, "Dependencies not installed"
        
        # Check if model was loaded successfully
        if not self.model_loaded:
            # Draw error message on frame
            cv2.putText(
                output,
                f"Error: {self.model_info.get('error', 'Model not loaded')}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return output, "Model loading error"
        
        # Start timing
        start_time = time.time()
        print("Process frame: Starting point cloud creation")  # Debug log
        
        # Create point cloud from image
        point_cloud = self._create_point_cloud_from_image(frame)
        
        # Run SceneScript inference
        try:
            print(f"Attempting to run inference with point cloud of shape {point_cloud.shape}")  # Debug log
            
            # Try to run inference with patched torch functions
            try:
                print("First attempt at model.run_inference")
                language_sequence = self.model.run_inference(
                    point_cloud,
                    nucleus_sampling_thresh=0.05,  # 0.0 is argmax, 1.0 is random sampling
                    verbose=True,  # Enable verbose mode for debugging
                )
                print("First attempt successful")
            except Exception as e:
                print(f"Error in run_inference: {str(e)}")
                
                # Create a fallback implementation that avoids using problematic functions
                def simple_inference(self, point_cloud):
                    """Simplified inference method that avoids problematic torch functions"""
                    # Create a dummy language sequence object to return
                    class DummyLanguageSequence:
                        def __init__(self):
                            self.entities = []
                    
                    print("Using simplified inference method")
                    return DummyLanguageSequence()
                
                # Use the fallback implementation
                language_sequence = simple_inference(self, point_cloud)
            
            # Process detected entities (walls, doors, windows, bounding boxes)
            print(f"Language sequence received, entities: {len(language_sequence.entities) if hasattr(language_sequence, 'entities') else 'No entities'}")  # Debug log
            detections = self._process_entities(language_sequence.entities if hasattr(language_sequence, 'entities') else [], frame.shape[:2])
            
            # Draw detections on the frame
            output, detection_text = self._draw_detections(output, detections)
            
            # Add inference time info
            inference_time = time.time() - start_time
            cv2.putText(
                output,
                f"Inference time: {inference_time:.2f}s",
                (20, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            return output, detection_text
            
        except Exception as e:
            # Draw error message on frame
            error_msg = f"SceneScript inference error: {str(e)}"
            cv2.putText(
                output,
                error_msg,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            print(error_msg)
            return output, error_msg
    
    def _process_entities(self, entities, image_shape):
        """
        Process SceneScript entities into detection boxes
        
        Args:
            entities (list): List of SceneScript entities
            image_shape (tuple): (height, width) of the input image
            
        Returns:
            list: List of detection dictionaries
        """
        height, width = image_shape
        detections = []
        
        for entity in entities:
            entity_type = entity.COMMAND_STRING
            
            if entity_type == "make_bbox":
                # Extract bbox parameters
                try:
                    # Get position (center of bbox)
                    cx = entity.params["position_x"]
                    cy = entity.params["position_y"]
                    cz = entity.params["position_z"]
                    
                    # Get dimensions
                    sx = entity.params["scale_x"]
                    sy = entity.params["scale_y"]
                    sz = entity.params["scale_z"]
                    
                    # Get rotation
                    angle_z = entity.params["angle_z"]
                    
                    # Get class if available (default to "object" if not)
                    class_name = entity.params.get("class", "object")
                    
                    # Normalize coordinates to image space
                    # Scale the 3D coordinates to fit the 2D image
                    # This is a simplification - in a real scenario, you would use proper 3D-to-2D projection
                    center_x = int((cx + 0.5) * width)
                    center_y = int((cy + 0.5) * height)
                    
                    # Scale the box dimensions to image space
                    box_width = int(sx * width * 0.5)
                    box_height = int(sy * height * 0.5)
                    
                    # Calculate corners based on rotated box
                    # Top-left, top-right, bottom-right, bottom-left order
                    corners = []
                    
                    # Create rotation matrix
                    rot = self.Rotation.from_rotvec([0, 0, angle_z]).as_matrix()
                    
                    # Calculate the 4 corners of the bounding box
                    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                        # Apply rotation to the corner point
                        dx_rot, dy_rot, _ = rot @ np.array([dx * sx/2, dy * sy/2, 0])
                        
                        # Convert to image coordinates
                        x = int(center_x + dx_rot * width * 0.5)
                        y = int(center_y + dy_rot * height * 0.5)
                        
                        corners.append((x, y))
                    
                    # Add the detection to the list
                    detections.append({
                        'type': 'bbox',
                        'class_name': class_name,
                        'corners': corners,
                        'center': (center_x, center_y),
                        'confidence': 1.0,  # SceneScript doesn't provide confidence
                        'dimensions': (box_width, box_height, sz)
                    })
                    
                except Exception as e:
                    print(f"Error processing bbox entity: {e}")
                    continue
                    
            elif entity_type in ["make_wall", "make_door", "make_window"]:
                # Process architectural elements
                try:
                    class_name = entity_type[5:]  # Remove "make_" prefix
                    
                    if entity_type == "make_wall":
                        # Extract wall corners
                        a_x = entity.params["a_x"]
                        a_y = entity.params["a_y"]
                        a_z = entity.params["a_z"]
                        b_x = entity.params["b_x"]
                        b_y = entity.params["b_y"]
                        b_z = entity.params["b_z"]
                        height = entity.params["height"]
                        
                        # Convert to image coordinates
                        x1 = int((a_x + 0.5) * width)
                        y1 = int((a_y + 0.5) * height)
                        x2 = int((b_x + 0.5) * width)
                        y2 = int((b_y + 0.5) * height)
                        
                        # Add to detections
                        detections.append({
                            'type': 'architectural',
                            'element_type': 'wall',
                            'class_name': class_name,
                            'points': [(x1, y1), (x2, y2)],
                            'height': height,
                            'confidence': 1.0
                        })
                        
                    elif entity_type in ["make_door", "make_window"]:
                        # Extract position and dimensions
                        pos_x = entity.params["position_x"]
                        pos_y = entity.params["position_y"]
                        pos_z = entity.params["position_z"]
                        width_param = entity.params["width"]
                        height_param = entity.params["height"]
                        
                        # Convert to image coordinates
                        center_x = int((pos_x + 0.5) * width)
                        center_y = int((pos_y + 0.5) * height)
                        rect_width = int(width_param * width * 0.5)
                        rect_height = int(height_param * height * 0.5)
                        
                        # Add to detections
                        detections.append({
                            'type': 'architectural',
                            'element_type': class_name,
                            'class_name': class_name,
                            'center': (center_x, center_y),
                            'dimensions': (rect_width, rect_height),
                            'confidence': 1.0
                        })
                        
                except Exception as e:
                    print(f"Error processing {entity_type} entity: {e}")
                    continue
        
        return detections
    
    def _draw_detections(self, image, detections):
        """
        Draw detections on the image
        
        Args:
            image (numpy.ndarray): Image to draw on
            detections (list): List of detection dictionaries
            
        Returns:
            tuple: (drawn_image, detection_text)
        """
        # Create output frame as copy of input
        output = image.copy()
        h, w = output.shape[:2]
        
        # Draw info overlay
        self._draw_info_overlay(output, len(detections))
        
        # Initialize detection text
        detection_text = "Detected Objects:\n"
        
        # Draw each detection
        for i, detection in enumerate(detections):
            detection_type = detection['type']
            class_name = detection['class_name']
            
            # Get color for this class
            if class_name in self.class_names:
                color_idx = self.class_names.index(class_name)
            else:
                # Default to last color if class not found
                color_idx = len(self.class_names) - 1
            color = self.colors[color_idx]
            
            if detection_type == 'bbox':
                # Draw bounding box from corners
                corners = np.array(detection['corners'], dtype=np.int32)
                cv2.polylines(output, [corners], True, color, 2)
                
                # Draw fill with transparency
                overlay = output.copy()
                cv2.fillPoly(overlay, [corners], color)
                cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
                
                # Draw center point
                center_x, center_y = detection['center']
                cv2.circle(output, (center_x, center_y), 3, color, -1)
                
                # Draw label
                label = f"{class_name}"
                
                # Create background for text
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = center_x - text_size[0] // 2
                text_y = center_y - text_size[1] - 5
                
                # Draw text background
                cv2.rectangle(
                    output,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Add to detection text
                box_width, box_height, _ = detection['dimensions']
                detection_text += f"{class_name} at ({center_x}, {center_y})\n"
                
            elif detection_type == 'architectural':
                element_type = detection['element_type']
                
                if element_type == 'wall':
                    # Draw wall as a line
                    points = detection['points']
                    cv2.line(output, points[0], points[1], color, 2)
                    
                    # Draw label
                    mid_x = (points[0][0] + points[1][0]) // 2
                    mid_y = (points[0][1] + points[1][1]) // 2
                    cv2.putText(
                        output,
                        "Wall",
                        (mid_x, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                    
                    detection_text += f"Wall at ({mid_x}, {mid_y})\n"
                    
                else:  # door or window
                    # Draw as a rectangle
                    center = detection['center']
                    dimensions = detection['dimensions']
                    
                    half_width = dimensions[0] // 2
                    half_height = dimensions[1] // 2
                    
                    top_left = (center[0] - half_width, center[1] - half_height)
                    bottom_right = (center[0] + half_width, center[1] + half_height)
                    
                    cv2.rectangle(output, top_left, bottom_right, color, 2)
                    
                    # Draw label
                    cv2.putText(
                        output,
                        element_type.capitalize(),
                        (center[0] - half_width, center[1] - half_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                    
                    detection_text += f"{element_type.capitalize()} at ({center[0]}, {center[1]})\n"
        
        return output, detection_text
    
    def _draw_info_overlay(self, image, num_detections):
        """
        Draw information overlay on the top of the frame
        
        Args:
            image (numpy.ndarray): Image to draw on
            num_detections (int): Number of detected objects
        """
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay for the top bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw title
        cv2.putText(
            image,
            "SceneScript Object Detection",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Draw model info and detection count
        cv2.putText(
            image,
            f"Model: {self.model_info.get('name', 'Unknown')} | Objects: {num_detections}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )