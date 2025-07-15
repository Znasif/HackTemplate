from .base_processor import BaseProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import cv2, os
from typing import Dict, Union, Tuple, List, Optional
from PIL import Image
import torch

class SceneProcessor(BaseProcessor):
    def __init__(self,
                 task_prompt='<DENSE_REGION_CAPTION>',
                 model_id='microsoft/Florence-2-large',
                 use_gpu=True,
                 min_confidence=0.5,
                 enable_layout_analysis=True):
        """
        Initialize Florence OCR processor
        
        Args:
            model_id (str): Florence model identifier
            use_gpu (bool): Whether to use GPU acceleration
            min_confidence (float): Minimum confidence threshold
            enable_layout_analysis (bool): Enable document layout analysis
        """
        super().__init__()
        self.task_prompt = task_prompt
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype='auto'
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.min_confidence = min_confidence
        self.enable_layout_analysis = enable_layout_analysis

    def _run_florence(self, image):
        """
        Run Florence on the image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            dict: results with regions and text
        """
        
        
        # Prepare inputs
        inputs = self.processor(
            text=self.task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.float16)

        # Generate results
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

        # Process output
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        return self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height)
        )
    
    def dense_caption(self, pil_image, output):
        # Run Florence OCR
        result = self._run_florence(pil_image)
        texts = ""
        # Process detected regions
        if result and self.task_prompt in result:
            ocr_result = result[self.task_prompt]
            
            for box, text in zip(ocr_result['bboxes'], ocr_result['labels']):
                # Convert quad box to numpy array
                x1, y1, x2, y2 = map(int, box)  # Convert to integers explicitly
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                texts += text + "\n"
                # Add text overlay
                cv2.putText(
                    output,
                    text,
                    (x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Specify fontFace
                    fontScale=1.0,  # Use fontScale for size (adjust as needed)
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,  # Correct parameter name
                    bottomLeftOrigin=False
                )
        print(texts)
        return output, texts

    def process_ocr(self, pil_image, output):
        # Run Florence OCR
        result = self._run_florence(pil_image)
        blank = np.zeros_like(output)
        
        # Initialize detections dictionary
        detections = {
            'text_regions': [],
            'layout': None,
            'structure': None
        }
        texts = ""
        # Process detected regions
        if result and self.task_prompt in result:
            ocr_result = result[self.task_prompt]
            
            for box, text in zip(ocr_result['quad_boxes'], ocr_result['labels']):
                # Convert quad box to numpy array
                box = np.array(box).reshape(-1, 2)
                texts += text + "\n"
                # Store detection info
                # detection = {
                #     'bbox': box.tolist(),
                #     'text': text,
                #     'confidence': 1.0  # Florence doesn't provide confidence scores
                #     #'orientation': self._detect_text_orientation(box)
                # }
                # detections['text_regions'].append(detection)
                
                # Draw bounding box
                
                #cv2.polylines(blank, [box_int], True, (0, 255, 0), 2)
                
                # Add text overlay
                text_x = int(box[0][0])
                text_y = int(box[0][1] - 10)
                cv2.putText(
                    blank,
                    text,
                    (text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontHeight=10,  # Use a proper font height instead of scale
                    color=(255, 255, 255),
                    thickness=1,
                    line_type=cv2.LINE_AA,  # Required argument
                    bottomLeftOrigin=False  # Add this argument
                )


        
        # Add layout analysis if enabled
        # if self.enable_layout_analysis and detections['text_regions']:
        #     detections['layout'] = self._analyze_layout(frame, detections['text_regions'])
        #     detections['structure'] = self._build_text_structure(detections['text_regions'])
            
        return blank, texts
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}


    def process_frame(self, frame):
        """
        Process frame using Florence to detect and recognize text
        
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
        
        # Convert OpenCV frame to PIL Image
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        
        if self.task_prompt == '<OCR_WITH_REGION>':
            # Run OCR with region detection
            return self.process_ocr(pil_image, output)
        else:
            return self.dense_caption(pil_image, output)

    # def _detect_text_orientation(self, box):
    #     """
    #     Detect text orientation based on bounding box
        
    #     Args:
    #         box (numpy.ndarray): Bounding box coordinates
            
    #     Returns:
    #         str: Orientation ('horizontal', 'vertical', or 'rotated')
    #     """
    #     width = np.linalg.norm(box[1] - box[0])
    #     height = np.linalg.norm(box[3] - box[0])
        
    #     if width > height * 1.5:
    #         return 'horizontal'
    #     elif height > width * 1.5:
    #         return 'vertical'
    #     else:
    #         return 'rotated'
    
    # def _analyze_layout(self, frame, text_regions):
    #     """
    #     Analyze document layout and group text regions
        
    #     Args:
    #         frame (numpy.ndarray): Input frame
    #         text_regions (list): Detected text regions
            
    #     Returns:
    #         dict: Layout analysis results
    #     """
    #     height, width = frame.shape[:2]
    #     layout = {
    #         'header': [],
    #         'body': [],
    #         'footer': [],
    #         'columns': []
    #     }
        
    #     # Define regions
    #     header_height = height * 0.2
    #     footer_height = height * 0.8
        
    #     # Group text regions based on position
    #     for region in text_regions:
    #         box = np.array(region['bbox'])
    #         center_y = np.mean(box[:, 1])
            
    #         if center_y < header_height:
    #             layout['header'].append(region)
    #         elif center_y > footer_height:
    #             layout['footer'].append(region)
    #         else:
    #             layout['body'].append(region)
        
    #     # Detect columns in body text
    #     if layout['body']:
    #         layout['columns'] = self._detect_columns(layout['body'])
            
    #     return layout
    
    # def _detect_columns(self, body_regions, gap_threshold=50):
    #     """
    #     Detect text columns in body regions
        
    #     Args:
    #         body_regions (list): Text regions in document body
    #         gap_threshold (int): Minimum gap to consider separate columns
            
    #     Returns:
    #         list: List of column definitions
    #     """
    #     if not body_regions:
    #         return []
            
    #     # Sort regions by x coordinate
    #     sorted_regions = sorted(body_regions, key=lambda r: np.mean(np.array(r['bbox'])[:, 0]))
        
    #     columns = []
    #     current_column = [sorted_regions[0]]
        
    #     # Group regions into columns based on horizontal gaps
    #     for region in sorted_regions[1:]:
    #         prev_region = current_column[-1]
    #         prev_box = np.array(prev_region['bbox'])
    #         curr_box = np.array(region['bbox'])
            
    #         gap = np.min(curr_box[:, 0]) - np.max(prev_box[:, 0])
            
    #         if gap > gap_threshold:
    #             columns.append(current_column)
    #             current_column = []
            
    #         current_column.append(region)
            
    #     if current_column:
    #         columns.append(current_column)
            
    #     return columns
    
    # def _build_text_structure(self, text_regions):
    #     """
    #     Build hierarchical text structure based on positions and sizes
        
    #     Args:
    #         text_regions (list): Detected text regions
            
    #     Returns:
    #         dict: Structured text hierarchy
    #     """
    #     # Sort regions by size and position
    #     sorted_regions = sorted(
    #         text_regions,
    #         key=lambda r: (
    #             -cv2.contourArea(np.array(r['bbox'])),  # Larger areas first
    #             np.mean(np.array(r['bbox'])[:, 1])      # Then by vertical position
    #         )
    #     )
        
    #     structure = {
    #         'title': None,
    #         'headings': [],
    #         'paragraphs': [],
    #         'tables': []
    #     }
        
    #     # Assign regions to structure based on characteristics
    #     for region in sorted_regions:
    #         box = np.array(region['bbox'])
    #         area = cv2.contourArea(box)
            
    #         if not structure['title'] and area > 1000:  # Adjust threshold as needed
    #             structure['title'] = region
    #         elif self._is_heading(region):
    #             structure['headings'].append(region)
    #         elif self._is_table_cell(region):
    #             structure['tables'].append(region)
    #         else:
    #             structure['paragraphs'].append(region)
                
    #     return structure
    
    # def _is_heading(self, region):
    #     """Check if text region appears to be a heading"""
    #     box = np.array(region['bbox'])
    #     width = np.linalg.norm(box[1] - box[0])
    #     height = np.linalg.norm(box[3] - box[0])
    #     return width < 500 and height > 20  # Adjust thresholds as needed
    
    # def _is_table_cell(self, region):
    #     """Check if text region appears to be part of a table"""
    #     box = np.array(region['bbox'])
    #     width = np.linalg.norm(box[1] - box[0])
    #     height = np.linalg.norm(box[3] - box[0])
    #     return width < 200 and height < 100  # Adjust thresholds as needed

processor = SceneProcessor()
app = processor.app