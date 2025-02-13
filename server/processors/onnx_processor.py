from .base_processor import BaseProcessor
import onnxruntime as ort
import numpy as np
import cv2

class ONNXProcessor(BaseProcessor):
    def __init__(self, model_path: str, input_name: str = None, confidence_threshold: float = 0.5):
        """
        Initialize ONNX model processor
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        
        # Get model input details
        model_inputs = self.session.get_inputs()
        self.input_name = input_name or model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        
        print(f"Model input name: {self.input_name}")
        print(f"Model input shape: {self.input_shape}")
        
        # For Faster R-CNN, input shape should be flexible
        # Only batch size and channels are fixed
        self.batch_size = self.input_shape[0]
        self.channels = self.input_shape[1] if len(self.input_shape) == 4 else 3
        
        # Target size for processing (can be adjusted)
        self.target_size = (800, 800)  # Common size for object detection
        print(f"Using target size: {self.target_size}")

    def preprocess_image(self, frame):
        """
        Preprocess the input frame for the model
        """
        # Resize image while maintaining aspect ratio
        height, width = frame.shape[:2]
        scale = min(self.target_size[0] / width, self.target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create a blank canvas of target size
        processed = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Calculate centering offsets
        y_offset = (self.target_size[1] - new_height) // 2
        x_offset = (self.target_size[0] - new_width) // 2
        
        # Place the resized image in the center
        processed[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Convert to RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        processed = processed.astype(np.float32) / 255.0
        
        # Add batch dimension and convert to NCHW format
        processed = np.expand_dims(processed, axis=0)
        processed = processed.transpose(0, 3, 1, 2)
        
        return processed

    def process_frame(self, frame):
        """
        Process a frame using the ONNX model
        """
        try:
            # Save original frame dimensions
            orig_height, orig_width = frame.shape[:2]
            
            # Preprocess the frame
            input_data = self.preprocess_image(frame)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # Create copy of frame for drawing
            result_frame = frame.copy()
            
            # Process detections
            # Assuming output format: boxes, scores, labels (might need adjustment)
            boxes = outputs[0]
            scores = outputs[1] if len(outputs) > 1 else None
            labels = outputs[2] if len(outputs) > 2 else None
            
            # Print shapes for debugging
            print(f"Output shapes: {[out.shape for out in outputs]}")
            
            # Draw detections
            if boxes is not None:
                for i, box in enumerate(boxes[0]):  # Process first batch only
                    if scores is not None and scores[0][i] < self.confidence_threshold:
                        continue
                        
                    # Get coordinates (assumed normalized)
                    x1, y1, x2, y2 = box[0:4]
                    
                    # Convert to pixel coordinates
                    x1 = int(x1 * orig_width)
                    y1 = int(y1 * orig_height)
                    x2 = int(x2 * orig_width)
                    y2 = int(y2 * orig_height)
                    
                    # Draw rectangle
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    score_text = f"{scores[0][i]:.2f}" if scores is not None else ""
                    label_text = f"Class {int(labels[0][i])}" if labels is not None else ""
                    text = f"{label_text} {score_text}".strip()
                    if text:
                        cv2.putText(result_frame, text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return result_frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            print(traceback.format_exc())
            return frame  # Return original frame on error