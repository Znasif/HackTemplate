#!/usr/bin/env python3
"""
Test script for FastVLM processor
"""
import cv2
import argparse
import os
import sys

# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processors.fastvlm_processor import FastVLMProcessor

def main():
    parser = argparse.ArgumentParser(description='Test FastVLM processor')
    parser.add_argument('--model-path', type=str, 
                        default="/home/znasif/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage3",
                        help='Path to FastVLM model')
    parser.add_argument('--prompt', type=str, 
                        default="Describe what you see in the image.",
                        help='Prompt for the model')
    parser.add_argument('--webcam', type=int, default=0, 
                        help='Webcam index to use')
    parser.add_argument('--model-size', type=str, default="0.5b",
                        choices=["0.5b", "1.5b", "7b"],
                        help='FastVLM model size variant')
    args = parser.parse_args()
    
    # Initialize processor
    processor = FastVLMProcessor(
        model_path=args.model_path,
        prompt=args.prompt,
        use_gpu=True,
        model_size=args.model_size
    )
    
    # Open webcam
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {args.webcam}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"FastVLM Processor Test")
    print(f"Model path: {args.model_path}")
    print(f"Model size: {args.model_size}")
    print(f"Prompt: {args.prompt}")
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
        
        # Add prompt to display
        cv2.putText(display_frame, f"Prompt: {args.prompt}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Process frame on demand
        if process_frame:
            output_frame, text_result = processor.process_frame(frame)
            last_result = text_result
            display_frame = output_frame
            process_frame = False
            print(f"Result: {text_result}")
        
        # Display instructions
        cv2.putText(display_frame, "Press 'p' to process frame, 'q' to quit", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display last result
        if last_result:
            # Break long text into multiple lines
            max_line_length = 60
            lines = []
            for i in range(0, len(last_result), max_line_length):
                lines.append(last_result[i:i+max_line_length])
            
            for i, line in enumerate(lines[:5]):  # Limit to 5 lines
                cv2.putText(display_frame, line, 
                            (10, 120 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('FastVLM Processor Test', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            process_frame = True
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
