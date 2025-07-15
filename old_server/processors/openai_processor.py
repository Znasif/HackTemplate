from .base_processor import BaseProcessor
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
import json
from openai import OpenAI

class ChatGPTProcessor(BaseProcessor):
    def __init__(self,
                 task_prompt='Describe this image in detail and identify any text visible in it.',
                 model_id='gpt-4o',
                 api_key=None,
                 min_confidence=0.5,
                 enable_layout_analysis=True):
        """
        Initialize ChatGPT processor using gpt-4o
        
        Args:
            task_prompt (str): Prompt to send to the model
            model_id (str): OpenAI model identifier
            api_key (str): OpenAI API key
            min_confidence (float): Minimum confidence threshold
            enable_layout_analysis (bool): Enable document layout analysis
        """
        super().__init__()
        self.task_prompt = task_prompt
        self.model_id = model_id
        
        if api_key is None:
            raise ValueError("OpenAI API key must be provided")
        
        self.client = OpenAI(api_key=api_key)
        self.min_confidence = min_confidence
        self.enable_layout_analysis = enable_layout_analysis
        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData("/home/znasif/vidServer/server/models/AtkinsonHyperlegible-Regular.ttf", 0)
        
        # Flag to track if a request is in progress
        self.is_processing = False
        # Store last result for reuse
        self.last_result = None
        self.last_frame = None
    
    def _encode_image(self, pil_image):
        """
        Encode PIL image to base64 for API request
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            str: Base64 encoded image
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _run_chatgpt(self, pil_image):
        """
        Run ChatGPT on the image
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            dict: results from ChatGPT
        """
        # Encode image to base64
        base64_image = self._encode_image(pil_image)
        
        # Prepare API request
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.task_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            
            # Extract text response
            result = response.choices[0].message.content
            return result
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None
    
    def _extract_text_regions(self, result):
        """
        Try to extract structured text region information from the model response
        
        Args:
            result (str): Model response text
            
        Returns:
            list: List of detected text regions or None if parsing fails
        """
        try:
            # Look for JSON-formatted text regions in the response
            # This is speculative - depends on prompt engineering and model output
            import re
            json_match = re.search(r'\{.*"regions".*\}', result, re.DOTALL)
            
            if json_match:
                regions_data = json.loads(json_match.group(0))
                if 'regions' in regions_data:
                    return regions_data['regions']
            
            return None
        except:
            return None
    
    def process_frame(self, frame):
        """
        Process frame using ChatGPT to detect and recognize text.
        Only processes if no other request is currently in progress.
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, text_result)
                - processed_frame (numpy.ndarray): Frame with information overlay
                - text_result (str): Text extracted or generated from the image
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Check if we're already processing a request
        if self.is_processing:
            # If we're still processing, render a "Processing..." indicator
            overlay = output.copy()
            cv2.rectangle(overlay, (10, 10), (output.shape[1] - 10, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
            
            # Add processing indicator
            self.font.putText(
                output,
                f"ChatGPT: {self.model_id} - Processing...",
                (20, 30),
                fontHeight=14,
                color=(255, 255, 255),
                thickness=1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
            
            # Return the current frame with processing indicator
            return output, "Processing..."
        
        # Not currently processing, so start a new request
        self.is_processing = True
        
        try:
            # Convert OpenCV frame to PIL Image
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            
            # Run ChatGPT model
            result = self._run_chatgpt(pil_image)
            
            # Store last result and frame for reuse
            self.last_result = result
            self.last_frame = output.copy()
            
            # Try to extract structured text regions (if prompt was configured for this)
            regions = self._extract_text_regions(result)
            
            # Add visualization if structured regions were detected
            if regions:
                for region in regions:
                    if 'bbox' in region and 'text' in region:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = region['bbox']
                        
                        # Draw rectangle
                        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add text overlay
                        self.font.putText(
                            output,
                            region['text'],
                            (int(x1), int(y1) - 5),
                            fontHeight=12,
                            color=(255, 255, 255),
                            thickness=1,
                            line_type=cv2.LINE_AA,
                            bottomLeftOrigin=False
                        )
            
            # Add general text overlay
            if result:
                # Add a semi-transparent background for text
                overlay = output.copy()
                cv2.rectangle(overlay, (10, 10), (output.shape[1] - 10, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
                
                # Add model info
                self.font.putText(
                    output,
                    f"ChatGPT: {self.model_id}",
                    (20, 30),
                    fontHeight=14,
                    color=(255, 255, 255),
                    thickness=1,
                    line_type=cv2.LINE_AA,
                    bottomLeftOrigin=False
                )
                
                # Add a small preview of the result
                result_preview = result[:50] + "..." if len(result) > 50 else result
                self.font.putText(
                    output,
                    result_preview,
                    (20, 50),
                    fontHeight=12,
                    color=(200, 200, 200),
                    thickness=1,
                    line_type=cv2.LINE_AA,
                    bottomLeftOrigin=False
                )
            
            return output, result
        
        finally:
            # Ensure the processing flag is reset even if an error occurs
            self.is_processing = False