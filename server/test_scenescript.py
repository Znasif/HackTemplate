#!/usr/bin/env python3
"""
Test script for SceneScript processor
"""
import cv2
import argparse
import os
import sys

# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.processors.scenescript_processor1 import SceneScriptProcessor

def main():
    parser = argparse.ArgumentParser(description='Test SceneScript processor')
    parser.add_argument('--model-path', type=str, 
                        default="/home/znasif/scenescript_model_non_manhattan_class_agnostic_model.ckpt",
                        help='Path to SceneScript model checkpoint')
    parser.add_argument('--webcam', type=int, default=0, 
                        help='Webcam index to use')
    parser.add_argument('--display-mode', type=str, default="all",
                        choices=["mesh", "wireframe", "bbox", "all"],
                        help='Visualization mode')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    args = parser.parse_args()
    
    # Initialize processor
    processor = SceneScriptProcessor(
        model_path=args.model_path,
        use_gpu=True,
        confidence_threshold=args.threshold,
        display_mode=args.display_mode
    )
    
    # Open webcam
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {args.webcam}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"SceneScript Processor Test")
    print(f"Model path: {args.model_path}")
    print(f"Display mode: {args.display_mode}")
    print(f"Confidence threshold: {args.threshold}")
    print("Press 'q' to quit, 'p' to process current frame")
    
    # Main loop
    process_frame = False
    last_result = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Add instructions to display
        cv2.putText(display_frame, "SceneScript: Press 'p' to process frame, 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process frame on demand
        if process_frame:
            output_frame, scene_description = processor.process_frame(frame)
            last_result = scene_description
            display_frame = output_frame
            process_frame = False
            print(f"Result:\n{scene_description}")
        
        # Display last result
        if last_result:
            # Display a condensed version on screen
            lines = last_result.split('\n')
            max_lines = min(5, len(lines))
            for i in range(max_lines):
                cv2.putText(display_frame, lines[i], 
                            (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('SceneScript Processor Test', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            process_frame = True
        elif key == ord('m'):
            # Toggle display mode
            modes = ["mesh", "wireframe", "bbox", "all"]
            current_idx = modes.index(processor.display_mode)
            next_idx = (current_idx + 1) % len(modes)
            processor.display_mode = modes[next_idx]
            print(f"Display mode changed to: {processor.display_mode}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
