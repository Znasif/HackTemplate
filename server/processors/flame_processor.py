import torch
import torchvision.transforms as TTR
import numpy as np
import cv2
import math
from .base_processor import BaseProcessor
import json

# Define the FlameProcessor extending BaseProcessor
class FlameProcessor(BaseProcessor):
    def __init__(self, jit_path: str = None, taxonomy_path: str = None):
        """
        Initialize the FlameProcessor by loading the pre-trained model and taxonomy.

        Args:
            jit_path: Path to the pre-trained model file (e.g., dms46_v1.jit).
            taxonomy_path: Path to the taxonomy JSON file.
        """
        super().__init__()
        
        # Define paths (replace with actual paths to your files)
        if jit_path is None:
            jit_path = '/home/znasif/vidServer/server/models/DMS46_v1.pt'
        if taxonomy_path is None:
            taxonomy_path = '/home/znasif/vidServer/server/models/taxonomy.json'
        
        print(f"DEBUG: Loading model from: {jit_path}")
        print(f"DEBUG: Loading taxonomy from: {taxonomy_path}")
        
        # Load the taxonomy for material names and color mapping
        try:
            with open(taxonomy_path, 'rb') as f:
                t = json.load(f)
            print(f"DEBUG: Taxonomy loaded successfully. Keys: {list(t.keys())}")
        except Exception as e:
            print(f"ERROR: Failed to load taxonomy: {e}")
            raise
        
        # List of label indices for the 46 materials (dms46)
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        print(f"DEBUG: DMS46 indices: {len(dms46)} materials")
        
        # Material names for dms46
        self.material_names = [t['names'][i] for i in dms46]
        print(f"DEBUG: Material names: {self.material_names}")
        
        # Original colormap for reference
        original_colormap = np.array([t['srgb_colormap'][i] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: Original colormap shape: {original_colormap.shape}")
        
        # Define flammability classes and map materials
        self.flammability_map = {
            1: 'A1',  # hide (animal skin)
            2: 'B/C',  # bone
            3: 'A1',  # brick
            4: 'D/E',  # cardboard
            5: 'D/E',  # carpet
            6: 'B/C',  # ceilingtile
            7: 'A1',  # ceramic
            8: 'B/C',  # chalkboard
            9: 'D/E',  # clutter
            10: 'A1',  # concrete
            11: 'B/C',  # cork
            12: 'A1',  # engineeredstone
            13: 'D/E',  # fabric
            15: 'F',   # fire
            16: 'D/E',  # foliage
            17: 'D/E',  # food
            18: 'D/E',  # fur
            19: 'D/E',  # gemstone (assumed non-flammable but not in A1)
            20: 'A2',  # glass
            21: 'D/E',  # hair
            23: 'D/E',  # icannottell
            24: 'D/E',  # leather
            26: 'A1',  # metal
            27: 'A2',  # mirror
            29: 'B/C',  # paint
            30: 'D/E',  # paper
            32: 'D/E',  # photograph
            33: 'D/E',  # clearplastic
            34: 'D/E',  # plastic
            35: 'D/E',  # rubber
            36: 'D/E',  # sand
            37: 'D/E',  # skin
            38: 'D/E',  # sky
            39: 'D/E',  # snow
            41: 'D/E',  # sponge
            43: 'A1',  # stone
            44: 'A1',  # polishedstone
            46: 'A1',  # tile
            47: 'B/C',  # wallpaper
            48: 'D/E',  # water
            49: 'D/E',  # wax
            50: 'B/C',  # whiteboard
            51: 'D/E',  # wicker
            52: 'D/E',  # wood
            53: 'D/E',  # treewood
            56: 'D/E'   # asphalt
        }
        
        print(f"DEBUG: Flammability mapping created for {len(self.flammability_map)} materials")
        
        # Define colors for each flammability class (RGB)
        try:
            self.class_colors = {
                'A1': original_colormap[dms46.index(26)],  # Yellow [225, 225, 0] for Non-combustible
                'A2': original_colormap[dms46.index(20)],  # Cyan [0, 137, 188] for Limited combustible
                'B/C': original_colormap[dms46.index(29)],  # Pink [225, 137, 188] for Flammable
                'D/E': original_colormap[dms46.index(52)],  # Dark red [137, 0, 0] for Normally flammable
                'F': original_colormap[dms46.index(15)]    # Light red [225, 188, 188] for Highly flammable
            }
            print(f"DEBUG: Class colors: {self.class_colors}")
        except Exception as e:
            print(f"ERROR: Failed to create class colors: {e}")
            raise
        
        # Create new colormap based on flammability classes
        self.srgb_colormap = np.array([self.class_colors[self.flammability_map[i]] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: SRGB colormap shape: {self.srgb_colormap.shape}")
        
        # Load the pre-trained model and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DEBUG: Using device: {self.device}")
        
        try:
            self.model = torch.jit.load(jit_path, map_location=self.device)
            self.model.eval()  # Set model to evaluation mode
            print("DEBUG: Model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            raise

    def apply_color(self, label_mask: np.ndarray) -> np.ndarray:
        """
        Apply the flammability-based color map to the label mask.
        
        Args:
            label_mask: 2D numpy array with integer label indices (0–45 for dms46).
        
        Returns:
            3D numpy array (height, width, 3) in BGR format.
        """
        print(f"DEBUG: apply_color - label_mask shape: {label_mask.shape}")
        print(f"DEBUG: apply_color - label_mask unique values: {np.unique(label_mask)}")
        print(f"DEBUG: apply_color - label_mask min/max: {label_mask.min()}/{label_mask.max()}")
        
        vis = np.take(self.srgb_colormap, label_mask, axis=0)  # Map labels to RGB colors
        print(f"DEBUG: apply_color - vis shape: {vis.shape}")
        return vis[..., ::-1]  # Convert RGB to BGR for OpenCV compatibility

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Process the input frame using material segmentation and return the processed frame
        and a text result listing detected flammability classes.
        
        Args:
            frame: Input image as a numpy array in BGR format.
        
        Returns:
            Tuple containing:
            - processed_frame: Concatenated original and segmented image in BGR format.
            - result_text: String listing detected flammability classes.
        """
        print(f"DEBUG: process_frame - Input frame shape: {frame.shape}")
        
        # Get original dimensions
        h, w = frame.shape[:2]
        print(f"DEBUG: Original dimensions: {h}x{w}")
        
        # Compute resize dimensions (smallest side to 512, maintain aspect ratio)
        scale = 512 / max(h, w)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        print(f"DEBUG: Resize scale: {scale}, new dimensions: {new_h}x{new_w}")
        
        # Resize the frame (keep in BGR for concatenation later)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"DEBUG: Resized frame shape: {resized_frame.shape}")
        
        # Convert to RGB for model input
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        print(f"DEBUG: RGB image shape: {img_rgb.shape}")
        
        # Convert to PyTorch tensor
        image = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float()  # To (C, H, W)
        print(f"DEBUG: Tensor shape before normalization: {image.shape}")
        print(f"DEBUG: Tensor min/max before normalization: {image.min()}/{image.max()}")
        
        # Normalize the image (values in [0, 255] range)
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        normalize = TTR.Normalize(mean, std)
        image = normalize(image)
        print(f"DEBUG: Tensor min/max after normalization: {image.min()}/{image.max()}")
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)
        print(f"DEBUG: Final tensor shape: {image.shape}")
        
        # Run inference
        print("DEBUG: Running model inference...")
        try:
            with torch.no_grad():
                model_output = self.model(image)
                print(f"DEBUG: Model output type: {type(model_output)}")
                print(f"DEBUG: Model output shape (if tensor): {model_output.shape if hasattr(model_output, 'shape') else 'N/A'}")
                
                # Check if output is a list/tuple
                if isinstance(model_output, (list, tuple)):
                    print(f"DEBUG: Model output is {type(model_output)} with {len(model_output)} elements")
                    for i, elem in enumerate(model_output):
                        print(f"DEBUG: Element {i} shape: {elem.shape if hasattr(elem, 'shape') else type(elem)}")
                
                prediction = model_output[0].detach().cpu().numpy()[0, 0]  # Shape: (H, W)
                print(f"DEBUG: Prediction shape: {prediction.shape}")
                print(f"DEBUG: Prediction dtype: {prediction.dtype}")
                print(f"DEBUG: Prediction min/max: {prediction.min()}/{prediction.max()}")
                print(f"DEBUG: Prediction unique values: {np.unique(prediction)}")
                
        except Exception as e:
            print(f"ERROR: Model inference failed: {e}")
            # Create dummy prediction for debugging
            prediction = np.zeros((new_h, new_w), dtype=np.int64)
            print("DEBUG: Using dummy prediction")
        
        prediction = prediction.astype(np.int64)  # Ensure integer indices
        print(f"DEBUG: Prediction after int conversion - unique values: {np.unique(prediction)}")
        
        # Visualize the prediction
        vis_bgr = self.apply_color(prediction)  # Shape: (new_h, new_w, 3) in BGR
        
        # Concatenate original and segmented images horizontally
        processed_frame = vis_bgr#np.concatenate((resized_frame, vis_bgr), axis=1)
        print(f"DEBUG: Processed frame shape: {processed_frame.shape}")
        
        # Identify unique flammability classes and generate text result
        unique_labels = np.unique(prediction)
        print(f"DEBUG: Unique labels in prediction: {unique_labels}")
        
        # Check the dms46 list for debugging
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        print(f"DEBUG: DMS46 length: {len(dms46)}")
        
        # Map labels to flammability classes with more detailed debugging
        detected_classes = []
        for label in unique_labels:
            print(f"DEBUG: Processing label: {label}")
            if 0 <= label < len(dms46):
                material_id = dms46[label]
                print(f"DEBUG: Label {label} maps to material ID {material_id}")
                if material_id in self.flammability_map:
                    flam_class = self.flammability_map[material_id]
                    print(f"DEBUG: Material ID {material_id} has flammability class {flam_class}")
                    detected_classes.append(flam_class)
                else:
                    print(f"DEBUG: Material ID {material_id} not found in flammability_map")
            else:
                print(f"DEBUG: Label {label} is out of range (0-{len(dms46)-1})")
        
        detected_classes = sorted(set(detected_classes))
        print(f"DEBUG: Final detected classes: {detected_classes}")
        
        result_text = "Detected flammability classes: " + ", ".join(detected_classes) if detected_classes else "No materials detected"
        print(f"DEBUG: Result text: {result_text}")
        
        return processed_frame, result_text

processor = FlameProcessor()
app = processor.app