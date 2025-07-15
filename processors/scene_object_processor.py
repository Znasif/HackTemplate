from .base_processor import BaseProcessor
from ultralytics import YOLO
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional

class YOLOProcessor(BaseProcessor):
    def __init__(self, model_path="sam2.1_b.pt"):
        """
        Initialize SAM processor with a model path
        
        Args:
            model_path (str): Path to the SAM model weights
        """
        super().__init__()
        self.model = YOLO(model_path)
        
    def process_frame(self, frame):
        """
        Process frame using SAM model to generate colored segmentation overlay and detections
        
        Args:
            frame (numpy.ndarray): Input frame to process
                
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame (numpy.ndarray): Frame with colored segmentation overlay
                - detections (list): List of detection dictionaries containing:
                    - bbox: [x1, y1, x2, y2]
                    - class_id: integer class ID
                    - class_name: string class name
                    - confidence: detection confidence score
        """
        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]
        
        # Run inference with SAM model
        results = self.model(frame)
        result = results[0]  # Get first image result
        
        # Create output frame as copy of input
        output = frame.copy()
        
        # Store detection information
        detections = []
        
        # Get the segmentation masks and boxes from results
        masks = result.masks
        boxes = result.boxes
        texts = ""
        
        if masks is not None and len(masks) > 0:
            # Process each detection
            for i in range(len(boxes)):
                # Get detection info
                box = boxes[i]
                mask_tensor = masks.data[i]
                
                # Extract detection details
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = result.names[class_id]
                
                texts += f"{class_name}\n"
                # Store detection info
                detection = {
                    'bbox': bbox.tolist(),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                }
                detections.append(detection)
                
                # Convert mask to numpy and resize
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(
                    mask.astype(float), 
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Generate random color for this segment
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0.5] = color
                
                # Blend with original frame
                alpha = 0.5  # Transparency factor
                mask_area = mask > 0.5
                output[mask_area] = cv2.addWeighted(frame[mask_area], 1-alpha, colored_mask[mask_area], alpha, 0)
                
                # Optionally draw bounding box and label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), color.tolist(), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(output, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        return output, texts
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}

    
    def process_frame_with_prompt(self, frame, prompt):
        """
        Process frame using SAM model with additional prompt
        
        Args:
            frame (numpy.ndarray): Input frame to process
            prompt (dict): Prompt dictionary containing points, boxes, or text
                
        Returns:
            numpy.ndarray: Original frame with colored segmentation overlay
        """
        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]
        
        # Run inference with prompt
        results = self.model(frame, **prompt)
        
        # Create output frame as copy of input
        output = frame.copy()
        
        # Get the segmentation masks from results
        masks = results[0].masks
        if masks is not None and len(masks) > 0:
            # Process each mask with different colors
            for i, mask_tensor in enumerate(masks.data):
                # Convert mask to numpy
                mask = mask_tensor.cpu().numpy()
                
                # Resize mask to match original frame dimensions
                mask = cv2.resize(
                    mask.astype(float), 
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Generate random color for this segment
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0.5] = color
                
                # Blend with original frame
                alpha = 0.5  # Transparency factor
                mask_area = mask > 0.5
                output[mask_area] = cv2.addWeighted(frame[mask_area], 1-alpha, colored_mask[mask_area], alpha, 0)
        
        return output, ""
    

processor = YOLOProcessor("./models/yolo11n-seg.pt")
app = processor.app