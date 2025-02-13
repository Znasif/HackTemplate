from .base_processor import BaseProcessor
import onnxruntime as ort
import numpy as np
import cv2
from typing import Tuple, List

class YOLOv8Processor(BaseProcessor):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize YOLO ONNX model processor
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
        # Initialize ONNX Runtime session with GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Set processing size (default YOLOv8 size)
        self.input_width = 640
        self.input_height = 640
        
        print(f"Model input name: {self.input_name}")
        print(f"Model input shape: {self.input_shape}")
        print(f"Model output names: {self.output_names}")
        
        # Initialize tracker
        self.tracks = {}
        self.track_id = 0

    def preprocess_image(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for YOLOv8 inference
        Returns:
            - Preprocessed image
            - Scale factor for width
            - Scale factor for height
        """
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Calculate scale factors
        scale_w = original_width / self.input_width
        scale_h = original_height / self.input_height
        
        # Resize image
        input_img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        input_img = input_img.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        input_img = input_img.transpose(2, 0, 1)
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, 0)
        
        return input_img, scale_w, scale_h

    def process_detections(self, output: np.ndarray, scale_w: float, scale_h: float) -> List[dict]:
        """
        Process YOLOv8 output to get detections
        """
        detections = []
        
        # YOLOv8 output format: [batch, num_boxes, xywh + confidence + num_classes]
        for box in output[0]:
            confidence = box[4]
            
            if confidence < self.confidence_threshold:
                continue
                
            # Get class with highest confidence
            class_id = np.argmax(box[5:])
            class_confidence = box[5 + class_id]
            
            # Convert xywh to xyxy
            x, y, w, h = box[0:4]
            x1 = (x - w/2) * scale_w
            y1 = (y - h/2) * scale_h
            x2 = (x + w/2) * scale_w
            y2 = (y + h/2) * scale_h
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_confidence': float(class_confidence)
            })
        
        return detections

    def update_tracks(self, detections: List[dict]) -> None:
        """
        Simple tracking using IoU matching
        """
        # Predict new locations of existing tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            # Add simple motion prediction here if needed
            track['age'] += 1
            if track['age'] > 10:  # Remove old tracks
                del self.tracks[track_id]
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for i, detection in enumerate(detections):
            best_iou = 0.3  # IoU threshold
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                iou = self.calculate_iou(detection['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update matched track
                self.tracks[best_track_id].update({
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'age': 0
                })
                matched_tracks.add(best_track_id)
                matched_detections.add(i)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.track_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class_id': detection['class_id'],
                    'age': 0
                }
                self.track_id += 1

    @staticmethod
    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """
        Calculate IoU between two boxes
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / float(area1 + area2 - intersection)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame using the YOLO model with tracking
        """
        try:
            # Preprocess image
            input_img, scale_w, scale_h = self.preprocess_image(frame)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_img})
            
            # Process detections
            detections = self.process_detections(outputs[0], scale_w, scale_h)
            
            # Update tracking
            self.update_tracks(detections)
            
            # Draw results
            result_frame = frame.copy()
            
            # Draw tracks
            for track_id, track in self.tracks.items():
                if track['age'] > 5:  # Skip old tracks
                    continue
                    
                x1, y1, x2, y2 = track['bbox']
                
                # Draw box
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID and confidence
                text = f"ID:{track_id} Conf:{track['confidence']:.2f}"
                cv2.putText(result_frame, text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return result_frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            print(traceback.format_exc())
            return frame  # Return original frame on error