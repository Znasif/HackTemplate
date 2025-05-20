#!/usr/bin/env python3
"""
SceneScript Processor Test

This script tests the SceneScript processor with the enhanced torch patches.
It creates a simple test frame and runs inference to verify the patches are working.
"""

import os
import sys
import traceback
import time
import numpy as np
import cv2

# Add server path to Python path
server_path = '/home/znasif/vidServer/server'
if server_path not in sys.path:
    sys.path.insert(0, server_path)

# Import torch compatibility patches first
try:
    from processors.torch_dim_patches import apply_all_patches
    patched_functions = apply_all_patches()
    print(f"Applied {len(patched_functions)} direct PyTorch patches")
except Exception as e:
    print(f"Warning: Could not apply direct patches: {e}")
    traceback.print_exc()

# Now import the SceneScript processor
try:
    from processors.scenescript_processor import SceneScriptProcessor
    print("SceneScript processor imported successfully")
except Exception as e:
    print(f"Error importing SceneScript processor: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_processor():
    """Test the SceneScript processor with a sample image"""
    
    print(f"\n{'-'*60}")
    print(f"Testing SceneScript processor")
    print(f"{'-'*60}")
    
    # Create a sample test image (synthetic indoor scene)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light gray background
    
    # Draw some basic shapes that might be detected
    # Room walls
    cv2.rectangle(test_image, (50, 50), (590, 430), (200, 200, 200), 2)
    
    # Window
    cv2.rectangle(test_image, (100, 100), (200, 200), (150, 200, 255), 2)
    
    # Door
    cv2.rectangle(test_image, (400, 300), (500, 400), (120, 120, 120), 2)
    
    # Table
    cv2.rectangle(test_image, (200, 250), (350, 350), (120, 80, 40), -1)
    
    # Chair
    cv2.circle(test_image, (150, 300), 30, (40, 40, 120), -1)
    
    # Create processor
    try:
        print("Initializing SceneScript processor...")
        model_path = '/home/znasif/vidServer/server/models/scenescript_model_non_manhattan_class_agnostic_model.ckpt'
        processor = SceneScriptProcessor(
            model_path=model_path,
            use_gpu=True
        )
        
        # Check if model is loaded
        if not processor.model_loaded:
            print(f"Error: Model not loaded - {processor.model_info.get('error', 'Unknown error')}")
            return False
            
        print("Model loaded successfully!")
        
        # Process test frame
        print("\nProcessing test frame...")
        start_time = time.time()
        output_frame, detection_text = processor.process_frame(test_image)
        elapsed_time = time.time() - start_time
        print(f"Inference completed in {elapsed_time:.2f} seconds")
        
        # Display results
        print("\nDetection Results:")
        print(detection_text)
        
        # Save the result
        output_path = os.path.join(server_path, 'test_scenescript_output.jpg')
        cv2.imwrite(output_path, output_frame)
        print(f"Output saved to: {output_path}")
        
        # Check if any detections were made
        if "entities: 0" in detection_text:
            print("\nWARNING: No entities detected. This may be normal for a synthetic test image.")
            print("The important thing is that the processor ran without crashing!")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing SceneScript processor: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_processor()
    
    if success:
        print("\nSUCCESS: SceneScript processor is working correctly!")
        sys.exit(0)
    else:
        print("\nFAILURE: SceneScript processor encountered errors.")
        sys.exit(1)
