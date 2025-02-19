from .base_processor import BaseProcessor
from paddleocr import PaddleOCR
import numpy as np
import cv2

class OCRProcessor(BaseProcessor):
    def __init__(self, 
                 languages=['en'],
                 use_angle_cls=True,
                 use_gpu=True,
                 enable_layout_analysis=True,
                 min_confidence=0.5):
        """
        Initialize OpenCV OCR processor using PaddleOCR
        
        Args:
            languages (list): List of language codes for OCR
            use_angle_cls (bool): Enable automatic rotation detection
            use_gpu (bool): Whether to use GPU acceleration
            enable_layout_analysis (bool): Enable document layout analysis
            min_confidence (float): Minimum confidence threshold
        """
        super().__init__()
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=languages[0],  # Primary language
            use_gpu=use_gpu,
            show_log=False,
            enable_mkldnn=True  # Enable Intel MKL-DNN acceleration
        )
        self.min_confidence = min_confidence
        self.enable_layout_analysis = enable_layout_analysis
        
    def process_frame(self, frame):
        """
        Process frame using PaddleOCR to detect and recognize text
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame (numpy.ndarray): Frame with detected text boxes
                - detections (dict): Dictionary containing:
                    - text_regions: List of text detection results
                    - layout: Document layout analysis (if enabled)
                    - structure: Structured text with relationships
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Run OCR detection and recognition
        result = self.ocr.ocr(frame, cls=True)
        
        # Initialize detections dictionary
        detections = {
            'text_regions': [],
            'layout': None,
            'structure': None
        }
        
        # Process detected regions
        if result:
            for line in result:
                if not line:
                    continue
                for box, (text, confidence) in line:
                    if confidence < self.min_confidence:
                        continue
                        
                    # Convert box points to integer coordinates
                    box = np.array(box, dtype=np.int32)
                    
                    # Store detection info
                    detection = {
                        'bbox': box.tolist(),
                        'text': text,
                        'confidence': confidence,
                        'orientation': self._detect_text_orientation(box)
                    }
                    detections['text_regions'].append(detection)
                    
                    # Draw bounding box
                    cv2.polylines(output, [box], True, (0, 255, 0), 2)
                    
                    # Calculate text position
                    text_x = int(box[0][0])
                    text_y = int(box[0][1] - 10)
                    
                    # Add text overlay with confidence
                    text_str = f"{text} ({confidence:.2f})"
                    cv2.putText(
                        output,
                        text_str,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
        
        # Add layout analysis if enabled
        if self.enable_layout_analysis and result:
            detections['layout'] = self._analyze_layout(frame, detections['text_regions'])
            detections['structure'] = self._build_text_structure(detections['text_regions'])
            
        return output
    
    def _detect_text_orientation(self, box):
        """
        Detect text orientation based on bounding box
        
        Args:
            box (numpy.ndarray): Bounding box coordinates
            
        Returns:
            str: Orientation ('horizontal', 'vertical', or 'rotated')
        """
        width = np.linalg.norm(box[1] - box[0])
        height = np.linalg.norm(box[3] - box[0])
        
        if width > height * 1.5:
            return 'horizontal'
        elif height > width * 1.5:
            return 'vertical'
        else:
            return 'rotated'
    
    def _analyze_layout(self, frame, text_regions):
        """
        Analyze document layout and group text regions
        
        Args:
            frame (numpy.ndarray): Input frame
            text_regions (list): Detected text regions
            
        Returns:
            dict: Layout analysis results
        """
        height, width = frame.shape[:2]
        layout = {
            'header': [],
            'body': [],
            'footer': [],
            'columns': []
        }
        
        # Define regions
        header_height = height * 0.2
        footer_height = height * 0.8
        
        # Group text regions based on position
        for region in text_regions:
            box = np.array(region['bbox'])
            center_y = np.mean(box[:, 1])
            
            if center_y < header_height:
                layout['header'].append(region)
            elif center_y > footer_height:
                layout['footer'].append(region)
            else:
                layout['body'].append(region)
        
        # Detect columns in body text
        if layout['body']:
            layout['columns'] = self._detect_columns(layout['body'])
            
        return layout
    
    def _detect_columns(self, body_regions, gap_threshold=50):
        """
        Detect text columns in body regions
        
        Args:
            body_regions (list): Text regions in document body
            gap_threshold (int): Minimum gap to consider separate columns
            
        Returns:
            list: List of column definitions
        """
        if not body_regions:
            return []
            
        # Sort regions by x coordinate
        sorted_regions = sorted(body_regions, key=lambda r: np.mean(np.array(r['bbox'])[:, 0]))
        
        columns = []
        current_column = [sorted_regions[0]]
        
        # Group regions into columns based on horizontal gaps
        for region in sorted_regions[1:]:
            prev_region = current_column[-1]
            prev_box = np.array(prev_region['bbox'])
            curr_box = np.array(region['bbox'])
            
            gap = np.min(curr_box[:, 0]) - np.max(prev_box[:, 0])
            
            if gap > gap_threshold:
                columns.append(current_column)
                current_column = []
            
            current_column.append(region)
            
        if current_column:
            columns.append(current_column)
            
        return columns
    
    def _build_text_structure(self, text_regions):
        """
        Build hierarchical text structure based on positions and sizes
        
        Args:
            text_regions (list): Detected text regions
            
        Returns:
            dict: Structured text hierarchy
        """
        # Sort regions by size and position
        sorted_regions = sorted(
            text_regions,
            key=lambda r: (
                -cv2.contourArea(np.array(r['bbox'])),  # Larger areas first
                np.mean(np.array(r['bbox'])[:, 1])      # Then by vertical position
            )
        )
        
        structure = {
            'title': None,
            'headings': [],
            'paragraphs': [],
            'tables': []
        }
        
        # Assign regions to structure based on characteristics
        for region in sorted_regions:
            box = np.array(region['bbox'])
            area = cv2.contourArea(box)
            
            if not structure['title'] and area > 1000:  # Adjust threshold as needed
                structure['title'] = region
            elif self._is_heading(region):
                structure['headings'].append(region)
            elif self._is_table_cell(region):
                structure['tables'].append(region)
            else:
                structure['paragraphs'].append(region)
                
        return structure
    
    def _is_heading(self, region):
        """Check if text region appears to be a heading"""
        box = np.array(region['bbox'])
        width = np.linalg.norm(box[1] - box[0])
        height = np.linalg.norm(box[3] - box[0])
        return width < 500 and height > 20  # Adjust thresholds as needed
    
    def _is_table_cell(self, region):
        """Check if text region appears to be part of a table"""
        # Simple heuristic based on region shape and nearby regions
        box = np.array(region['bbox'])
        width = np.linalg.norm(box[1] - box[0])
        height = np.linalg.norm(box[3] - box[0])
        return width < 200 and height < 100  # Adjust thresholds as needed