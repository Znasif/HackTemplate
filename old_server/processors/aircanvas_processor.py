"""
AirCanvas Gesture Processor

This processor implements gesture-driven writing functionality
by tracking hand movements in the frames provided by the HackTemplate server
using MediaPipe and OpenCV.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, Any, Tuple, List, Optional
import base64
from processors.base_processor import BaseProcessor

class AirCanvasProcessor(BaseProcessor):
    """
    A processor that implements air canvas functionality for gesture-driven writing.
    This processor extends the BaseProcessor class from the HackTemplate framework.
    """
    
    def __init__(self):
        """Initialize the AirCanvas processor with default settings."""
        super().__init__()
        self.name = "air_canvas"
        self.description = "Gesture-driven writing processor using hand tracking"
        
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the hand detector
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Canvas and drawing state
        self.canvas = None
        self.prev_position = None
        self.frame_dimensions = None
        
        # Drawing settings
        self.draw_color = (0, 0, 255)  # Red by default
        self.thickness = 5
        
        # Gesture state
        self.drawing_active = False
        
        # Drawing mode
        self.drawing_mode = "free"  # "free" or "character"
        
        # For character recognition mode
        self.character_canvas = None
        self.last_drawing_time = None
        self.character_timeout = 2.0  # seconds
    
    def initialize_canvas(self, frame_shape):
        """Initialize or resize the canvas to match the frame dimensions."""
        if self.canvas is None or self.frame_dimensions != (frame_shape[1], frame_shape[0]):
            self.frame_dimensions = (frame_shape[1], frame_shape[0])
            self.canvas = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
            if self.drawing_mode == "character":
                self.character_canvas = np.zeros_like(self.canvas)
    
    def clear_canvas(self) -> None:
        """Clear the drawing canvas."""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
            if self.character_canvas is not None:
                self.character_canvas = np.zeros_like(self.canvas)
    
    def set_color(self, color: Tuple[int, int, int]) -> None:
        """Set the drawing color."""
        self.draw_color = color
    
    def set_thickness(self, thickness: int) -> None:
        """Set the drawing line thickness."""
        self.thickness = max(1, min(thickness, 20))  # Limit between 1-20px
    
    def set_mode(self, mode: str) -> None:
        """Set the drawing mode (free or character)."""
        if mode in ["free", "character"]:
            self.drawing_mode = mode
            # Reset canvas for character mode
            if mode == "character" and self.canvas is not None:
                self.character_canvas = np.zeros_like(self.canvas)
                self.last_drawing_time = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming data based on the command.
        Implementation of the abstract method from BaseProcessor.
        
        Args:
            data (Dict[str, Any]): The input data containing frames and commands
            
        Returns:
            Dict[str, Any]: The response with processed frames and data
        """
        command = data.get("command", "")
        
        if command == "process_frame":
            # Process a frame from the camera
            frame = data.get("frame")
            if frame is None:
                return {"status": "error", "message": "No frame provided"}
            try:
                # Process the frame and get the result
                processed_frame, detection_text = self.process_frame(frame)
                
                return {
                    "status": "success",
                    "processed_frame": processed_frame,
                    "detection_text": detection_text
                }
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                return {
                    "status": "error",
                    "message": f"Error processing frame: {str(e)}",
                    "error_type": str(type(e)),
                    "traceback": error_traceback
                }
        
        elif command == "clear":
            self.clear_canvas()
            return {"status": "success", "message": "Canvas cleared"}
        
        elif command == "set_color":
            color = data.get("color", (0, 0, 255))
            self.set_color(color)
            return {"status": "success", "message": f"Color set to {color}"}
        
        elif command == "set_thickness":
            thickness = data.get("thickness", 5)
            self.set_thickness(thickness)
            return {"status": "success", "message": f"Thickness set to {thickness}"}
        
        elif command == "set_mode":
            mode = data.get("mode", "free")
            self.set_mode(mode)
            return {"status": "success", "message": f"Mode set to {mode}"}
        
        elif command == "get_canvas":
            return {
                "status": "success", 
                "image": self.get_canvas_base64()
            }
        
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
        
    def process_frame(self, frame):
        """
        Process a single frame to detect hand gestures and update the canvas.
        
        Args:
            frame: The input camera frame to process
            
        Returns:
            Tuple[np.ndarray, str]: The processed frame with visualization and detection text
        """ 
        # Initialize canvas if not already done
        self.initialize_canvas(frame.shape)
        
        # Make a copy of the frame for drawing on
        display_frame = frame.copy()
        detection_text = ""
        
        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks and handle gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the display frame
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get the position of the index finger tip
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Get the middle and thumb positions to determine if drawing is active
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Calculate the distance between thumb and middle finger
                # If they are close, activate drawing
                thumb_middle_distance = self._calculate_distance(thumb_tip, middle_tip)
                self.drawing_active = thumb_middle_distance < 0.07  # Adjust threshold as needed
                
                # Calculate the absolute position of the index finger
                h, w, _ = frame.shape
                index_position = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Draw a circle at the index finger position
                circle_color = (0, 255, 0) if self.drawing_active else (0, 0, 255)
                cv2.circle(display_frame, index_position, 10, circle_color, -1)
                
                # Handle drawing based on the mode
                if self.drawing_active:
                    self.last_drawing_time = time.time()
                    if self.drawing_mode == "free":
                        self._handle_free_drawing(index_position)
                        detection_text = "Drawing"
                    elif self.drawing_mode == "character":
                        self._handle_character_drawing(index_position)
                        detection_text = "Drawing character"
                else:
                    # Reset the previous position if not drawing
                    self.prev_position = None
                    detection_text = "Hand detected"
                    
                    # In character mode, check if we should process the character
                    if (self.drawing_mode == "character" and 
                        self.last_drawing_time is not None and 
                        time.time() - self.last_drawing_time > self.character_timeout):
                        self._process_character()
                        self.last_drawing_time = None
                        detection_text = "Character processed"
        else:
            # If no hands detected for more than the timeout in character mode,
            # and we were recently drawing, process the character
            if (self.drawing_mode == "character" and 
                self.last_drawing_time is not None and 
                time.time() - self.last_drawing_time > self.character_timeout):
                self._process_character()
                self.last_drawing_time = None
                detection_text = "Character processed"
            else:
                detection_text = "No hand detected"
        
        # Blend the canvas with the display frame if canvas exists
        if self.canvas is not None and np.any(self.canvas > 0):
            # Simply blend the entire frames with addWeighted
            result_frame = cv2.addWeighted(display_frame, 0.7, self.canvas, 0.3, 0)
            return result_frame, detection_text
        
        return display_frame, detection_text
    
    def get_canvas_image(self) -> np.ndarray:
        """Return the current canvas as an image."""
        if self.canvas is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.canvas.copy()
    
    def get_canvas_base64(self) -> str:
        """Return the current canvas as a base64 encoded image."""
        if self.canvas is None:
            return ""
        
        success, buffer = cv2.imencode('.png', self.canvas)
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        return ""
    
    def _handle_free_drawing(self, position: Tuple[int, int]) -> None:
        """Handle free-form drawing on the canvas."""
        if self.prev_position is not None:
            cv2.line(self.canvas, self.prev_position, position, self.draw_color, self.thickness)
        self.prev_position = position
    
    def _handle_character_drawing(self, position: Tuple[int, int]) -> None:
        """Handle character drawing mode."""
        if self.character_canvas is None and self.canvas is not None:
            self.character_canvas = np.zeros_like(self.canvas)
        
        if self.prev_position is not None and self.character_canvas is not None:
            cv2.line(self.character_canvas, self.prev_position, position, (255, 255, 255), self.thickness)
        self.prev_position = position
    
    def _process_character(self) -> None:
        """Process the drawn character (placeholder for actual recognition)."""
        # For demonstration, just transfer the character to the main canvas
        if self.character_canvas is not None and self.canvas is not None:
            # Transfer the character to the main canvas
            mask = self.character_canvas > 0
            self.canvas[mask] = self.character_canvas[mask]
            
            # Reset the character canvas
            self.character_canvas = np.zeros_like(self.canvas)
    
    def _calculate_distance(self, landmark1, landmark2) -> float:
        """Calculate the Euclidean distance between two landmarks."""
        return ((landmark1.x - landmark2.x) ** 2 + 
                (landmark1.y - landmark2.y) ** 2 + 
                (landmark1.z - landmark2.z) ** 2) ** 0.5
